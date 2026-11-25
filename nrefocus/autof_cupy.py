"""CuPy-specific autofocus helpers with controllable CUDA stream use."""
from concurrent.futures import ThreadPoolExecutor

from ._ndarray_backend import xp
from . import iface


def autofocus_stack_cupy(fieldstack, nm, res, ival, roi=None,
                         metric="average gradient", minimizer="legacy_cupy",
                         minimizer_kwargs=None, padding=True,
                         num_streams=4, copy=True):
    """Autofocus each 2D frame in a stack using CuPy CUDA streams.

    Each frame gets an independent focus search. Up to ``num_streams`` frame
    searches are submitted concurrently, each on its own non-blocking CUDA
    stream. The batched CuPy minimizer evaluates candidate distances together.

    Parameters
    ----------
    fieldstack : (n, y, x) array
        Stack of complex 2D fields. NumPy input is copied to the active GPU.
    nm : float
        Refractive index of the medium.
    res : float
        Wavelength in pixels.
    ival : tuple of float
        Search interval in pixels.
    roi : ROI accepted by :meth:`Refocus.autofocus`, optional
        Region used to evaluate each frame's metric.
    metric : str
        Metric name supported by nrefocus.
    minimizer : str
        ``"legacy_cupy"`` evaluates candidate distances in batches.
        ``"lmfit_cupy"`` runs a one-parameter LM refinement on the GPU.
        ``"legacy"`` retains the serial candidate loop.
        ``"lmfit"`` runs lmfit on the CPU and evaluates each metric on the GPU.
    minimizer_kwargs : dict, optional
        Keyword arguments passed to the selected minimizer.
        The batched ``"legacy_cupy"`` minimizer supports ``coarse_acc`` and
        ``fine_acc``.
        ``"lmfit_cupy"`` supports ``coarse_step``, ``max_iter``, ``ftol``,
        ``xtol``, ``fd_step``, and ``initial_damping``.
        ``"lmfit"`` accepts options through its ``lmfitkw`` argument.
    padding : bool
        Apply spatial padding before propagation.
    num_streams : int
        Maximum number of independent frame searches to submit concurrently.
        For small jobs, keep this at 1, as using more streams will increase
        overhead.
    copy : bool
        Copy the input stack to device memory when it is already a CuPy array.

    Returns
    -------
    distances : list of float or cupy.ndarray
        One optimal distance per input frame, in pixels. The ``"lmfit"``
        minimizer returns Python floats; the other minimizers return
        zero-dimensional CuPy arrays.
    focused_stack : cupy.ndarray
        Refocused stack, in input frame order.

    """
    if not xp.is_cupy():
        raise RuntimeError(
            "Set the ndarray backend to 'cupy' before calling "
            "autofocus_stack_cupy().")
    if len(fieldstack.shape) != 3:
        raise ValueError("`fieldstack` must have shape (n, y, x).")
    if num_streams < 1:
        raise ValueError("`num_streams` must be at least 1.")
    supported_minimizers = ("legacy", "legacy_cupy", "lmfit", "lmfit_cupy")
    if minimizer not in supported_minimizers:
        raise ValueError("CuPy autofocus currently supports "
                         f"the minimizers: {supported_minimizers}. "
                         f"You tried to use '{minimizer}'.")

    # Transfer the stack once, then give each stream a view of its own frame.
    gpu_stack = xp.array(fieldstack, copy=copy)
    streams = [xp.cuda.Stream(non_blocking=True)
               for _ in range(min(num_streams, gpu_stack.shape[0]))]
    kwargs = {} if minimizer_kwargs is None else dict(minimizer_kwargs)

    def autofocus_frame(index):
        with streams[index % len(streams)]:
            rf = iface.RefocusCupy(
                field=gpu_stack[index],
                wavelength=res,
                pixel_size=1,
                medium_index=nm,
                distance=0,
                kernel="helmholtz",
                padding=padding,
            )
            return rf.autofocus(
                metric=metric,
                minimizer=minimizer,
                interval=xp.asarray(ival),
                roi=roi,
                minimizer_kwargs=kwargs,
                ret_grid=False,
                ret_field=True,
            )

    with ThreadPoolExecutor(max_workers=len(streams)) as executor:
        results = list(executor.map(autofocus_frame,
                                    range(gpu_stack.shape[0])))

    # Worker streams are non-blocking relative to the main thread's default
    # stream; finish them before stacking their outputs on the default stream.
    for stream in streams:
        stream.synchronize()

    distances = [result[0] for result in results]
    focused_stack = xp.stack([result[1] for result in results], axis=0)
    return distances, focused_stack
