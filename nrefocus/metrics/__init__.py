from .mt_avg_grad import metric_average_gradient
from .mt_rms_contrast import metric_rms_contrast
from .mt_spectrum import metric_spectrum
from .mt_std_grad import metric_std_gradient
from .mt_med_grad import metric_med_gradient


#: Available metrics
METRICS = {
    "average gradient": metric_average_gradient,
    "rms contrast": metric_rms_contrast,
    "spectrum": metric_spectrum,
    "std gradient": metric_std_gradient,
    "med gradient": metric_med_gradient,
}
