from numpy import asarray
import numpy as np
import math
from scipy.signal import zpk2tf
from Code.Heart_disease_prediction_Using_PJM_DJRNN_Testing import global_input_ecg_signal
print("Existing Bessel filter was executing...")
try:
    from mpmath import mp
    mpmath_available = True
except ImportError:
    mpmath_available = False
def bessel_poly(n, reverse=False):
    out = []
    for k in range(n + 1):
        num = math.factorial(2*n - k, exact=True)
        den = 2**(n - k) * (math.factorial(k, exact=True) *
                            math.factorial(n - k, exact=True))
        out.append(num // den)

    if reverse:
        return list(reversed(out))
    else:
        return out

def normfactor(a):

    def G(w):

        return abs(a[-1]/mp.polyval(a, 1j*w))

    def cutoff(w):

        return G(w) - 1/mp.sqrt(2)

    return mp.findroot(cutoff, 1.5)

def _roots(a):


    N = (len(a) - 1)//2  # Order of the filter

    if mpmath_available:

        mp.dps = 150

        p, err = mp.polyroots(a, maxsteps=1000, error=True)
        if err > 1e-32:
            raise ValueError("Filter cannot be accurately computed "
                             "for order %s" % N)
        p = asarray(p).astype(complex)
    else:
        p = np.roots(a)
        if N > 25:
            # Bessel and Legendre filters seem to fail above N = 25
            raise ValueError("Filter cannot be accurately computed "
                             "for order %s" % N)
    return p


def besselap(N, norm='phase'):

    if N == 0:
        return asarray([]), asarray([]), 1

    # Find delay-normalized Bessel filter poles
    a = bessel_poly(N, reverse=True)
    p = _roots(a)

    if norm == 'delay':
        # Normalized for group delay of 1
        k = a[-1]
    elif norm == 'phase':
        # Phase-matched (1/2 max phase shift at 1 rad/sec)
        # Asymptotes are same as Butterworth filter
        p *= 1 / a[-1]**(1/N)
        k = 1
    elif norm == 'mag':
        # -3 dB magnitude point is at 1 rad/sec
        p *= 1 / normfactor(a)
        k = float(normfactor(a)**-N * a[-1])
    else:
        raise ValueError('normalization not understood')

    z = []
    p = p.astype(complex)

    return asarray(z), asarray(p), k


def tests():
    from scipy import signal
    from numpy.testing import (assert_allclose, assert_array_equal,
                               assert_array_almost_equal)

    bessels = {
        0: [1],
        1: [1, 1],
        2: [3, 3, 1],
        3: [15, 15, 6, 1],
        4: [105, 105, 45, 10, 1],
        5: [945, 945, 420, 105, 15, 1]
        }

    for N in range(6):
        assert_array_equal(bessels[N], bessel_poly(N))
        assert_array_equal(bessels[N][::-1],
                           bessel_poly(N, reverse=True))

    a = bessel_poly(2, reverse=True)
    # Compare with symbolic result
    assert_allclose(float(normfactor(a)),
                    (np.sqrt(3)*np.sqrt(np.sqrt(5)-1)) / np.sqrt(2))

    Bond = {2: 1.36165412871613,
            3: 1.75567236868121,
            4: 2.11391767490422,
            5: 2.42741070215263,
            6: 2.70339506120292,
            7: 2.95172214703872,
            8: 3.17961723751065,
            9: 3.39169313891166,
            10: 3.59098059456916,
            }

    for N in range(2, 11):
        a = bessel_poly(N, reverse=True)
        assert_allclose(float(normfactor(a)), Bond[N])


    bond_b = 10395
    bond_a = [1, 21, 210, 1260, 4725, 10395, 10395]
    b, a = zpk2tf(*besselap(6, 'delay'))
    assert_allclose(bond_b, b)
    assert_allclose(bond_a, a)

    bond_poles = {
        1: [-1.0000000000],
        2: [-1.5000000000 + 0.8660254038j],
        3: [-1.8389073227 + 1.7543809598j, -2.3221853546],
        4: [-2.1037893972 + 2.6574180419j, -2.8962106028 + 0.8672341289j],
        5: [-2.3246743032 + 3.5710229203j, -3.3519563992 + 1.7426614162j,
            -3.6467385953],
        6: [-2.5159322478 + 4.4926729537j, -3.7357083563 + 2.6262723114j,
            -4.2483593959 + 0.8675096732j],
        7: [-2.6856768789 + 5.4206941307j, -4.0701391636 + 3.5171740477j,
            -4.7582905282 + 1.7392860611j, -4.9717868585],
        8: [-2.8389839489 + 6.3539112986j, -4.3682892172 + 4.4144425005j,
            -5.2048407906 + 2.6161751526j, -5.5878860433 + 0.8676144454j],
        9: [-2.9792607982 + 7.2914636883j, -4.6384398872 + 5.3172716754j,
            -5.6044218195 + 3.4981569179j, -6.1293679043 + 1.7378483835j,
            -6.2970191817],
        10: [-3.1089162336 + 8.2326994591j, -4.8862195669 + 6.2249854825j,
             -5.9675283286 + 4.3849471889j, -6.6152909655 + 2.6115679208j,
             -6.9220449054 + 0.8676651955j]
        }


    bond_poles = {
        1: [-1.0000000000],
        2: [-1.1016013306 + 0.6360098248j],
        3: [-1.0474091610 + 0.9992644363j, -1.3226757999],
        4: [-0.9952087644 + 1.2571057395j, -1.3700678306 + 0.4102497175j],
        5: [-0.9576765486 + 1.4711243207j, -1.3808773259 + 0.7179095876j,
            -1.5023162714],
        6: [-0.9306565229 + 1.6618632689j, -1.3818580976 + 0.9714718907j,
            -1.5714904036 + 0.3208963742j],
        7: [-0.9098677806 + 1.8364513530j, -1.3789032168 + 1.1915667778j,
            -1.6120387662 + 0.5892445069j, -1.6843681793],
        8: [-0.8928697188 + 1.9983258436j, -1.3738412176 + 1.3883565759j,
            -1.6369394181 + 0.8227956251j, -1.7574084004 + 0.2728675751j],
        9: [-0.8783992762 + 2.1498005243j, -1.3675883098 + 1.5677337122j,
            -1.6523964846 + 1.0313895670j, -1.8071705350 + 0.5123837306j,
            -1.8566005012],
        10: [-0.8657569017 + 2.2926048310j, -1.3606922784 + 1.7335057427j,
             -1.6618102414 + 1.2211002186j, -1.8421962445 + 0.7272575978j,
             -1.9276196914 + 0.2416234710j]
        }

    for N in range(26):
        assert_allclose(sorted(signal.besselap(N)[1]), sorted(besselap(N)[1]))

    a = [1, 1, 1/3]
    b2, a2 = zpk2tf(*besselap(2, 'delay'))
    assert_allclose(a[::-1], a2/b2)

    a = [1, 1, 2/5, 1/15]
    b2, a2 = zpk2tf(*besselap(3, 'delay'))
    assert_allclose(a[::-1], a2/b2)

    a = [1, 1, 9/21, 2/21, 1/105]
    b2, a2 = zpk2tf(*besselap(4, 'delay'))
    assert_allclose(a[::-1], a2/b2)

    a = [1, np.sqrt(3), 1]
    b2, a2 = zpk2tf(*besselap(2, 'phase'))
    assert_allclose(a[::-1], a2/b2)

    a = [1, 2.481, 2.463, 1.018]
    b2, a2 = zpk2tf(*besselap(3, 'phase'))
    assert_array_almost_equal(a[::-1], a2/b2, decimal=1)


    a = [1, 3.240, 4.5, 3.240, 1.050]
    b2, a2 = zpk2tf(*besselap(4, 'phase'))
    assert_array_almost_equal(a[::-1], a2/b2, decimal=1)

    N, scale = 2, 1.272
    scale2 = besselap(N, 'mag')[1] / besselap(N, 'phase')[1]
    assert_array_almost_equal(scale, scale2, decimal=3)

    N, scale = 3, 1.413
    scale2 = besselap(N, 'mag')[1] / besselap(N, 'phase')[1]
    assert_array_almost_equal(scale, scale2, decimal=2)

    N, scale = 4, 1.533
    scale2 = besselap(N, 'mag')[1] / besselap(N, 'phase')[1]
    assert_array_almost_equal(scale, scale2, decimal=1)
print("Existing Bessel filter was executed successfully...")