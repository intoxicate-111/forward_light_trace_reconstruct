"""Small numerical continuation controls; complex values never reach rendering."""
import numpy as np


def continue_root(path, initial, equation, spatial, parameter, steps=512):
    """RK4 implicit tangent predictor followed by full-equation Newton correction.

    path(t) returns parameter and dparameter/dt. No gradient penalty/barrier.
    Root arrays allow independent normal-line samples in one continuation.
    """
    root = np.asarray(initial, dtype=np.complex128).copy()
    roots = [root.copy()]; residuals = []; jacobians = []; derivatives = []; parameters = []; tangents = []
    dt = 1/steps
    def tangent(t, x):
        c, dc = path(t)
        return -parameter(x, c)/spatial(x, c)*dc
    for i in range(steps+1):
        t = i/steps; c, dc = path(t)
        residuals.append(float(np.max(np.abs(equation(root, c)))))
        jacobians.append(float(np.min(np.abs(spatial(root, c)))))
        derivatives.append(float(np.max(np.abs(parameter(root, c)/spatial(root, c)))))
        tangents.append(-parameter(root, c)/spatial(root, c)*dc)
        parameters.append(c)
        if i == steps:
            break
        k1 = tangent(t, root); k2 = tangent(t+dt/2, root+dt*k1/2)
        k3 = tangent(t+dt/2, root+dt*k2/2); k4 = tangent(t+dt, root+dt*k3)
        root = root+dt*(k1+2*k2+2*k3+k4)/6
        next_c, _ = path(t+dt)
        for _ in range(5):
            root = root-equation(root, next_c)/spatial(root, next_c)
        if not np.isfinite(root).all():
            raise RuntimeError('nonfinite continuation root')
        roots.append(root.copy())
    sampled = np.asarray(roots); derivative_fd = (sampled[2:]-sampled[:-2])/(2*dt)
    tangent_array = np.asarray(tangents)[1:-1]
    error = float(np.linalg.norm(derivative_fd-tangent_array)/max(np.linalg.norm(tangent_array), 1e-30))
    return dict(roots=sampled, parameters=np.asarray(parameters), residuals=residuals, tangent_fd_relative_l2=error,
                spatial_jacobian_min=jacobians, implicit_derivative_max=derivatives)


def summary(result):
    return dict(max_residual=max(result['residuals']), min_spatial_jacobian=min(result['spatial_jacobian_min']),
                path_tangent_fd_relative_l2=result['tangent_fd_relative_l2'],
                max_implicit_derivative=max(result['implicit_derivative_max']),
                start_real=np.real(result['roots'][0]).tolist(), start_imag=np.imag(result['roots'][0]).tolist(),
                final_real=np.real(result['roots'][-1]).tolist(), final_imag=np.imag(result['roots'][-1]).tolist(),
                endpoint_imag_max=float(np.abs(result['roots'][-1].imag).max()))


def scalar_control():
    def path(t):
        mu = np.exp(2j*np.pi*t)
        return mu, 2j*np.pi*mu
    result = continue_root(path, np.array([1.]), lambda z, c: z*z-c,
                           lambda z, c: 2*z, lambda z, c: -np.ones_like(z))
    real = [dict(mu=float(mu), spatial_gradient=float(2*np.sqrt(mu)),
                 implicit_derivative=float(1/(2*np.sqrt(mu)))) for mu in np.logspace(0, -14, 15)]
    out = summary(result)
    out['min_distance_to_discriminant'] = float(np.abs(result['parameters']).min())
    out.update(real_path=real, passed=bool(abs(result['roots'][-1, 0]+1) < 1e-12 and out['max_residual'] < 1e-12
                                         and out['min_spatial_jacobian'] > 1.99))
    return out, result


def sphere_torus_control():
    def path(t):
        turn = .2*np.exp(1j*np.pi*(1-t))
        return 1+turn, -1j*np.pi*turn
    results = {}; rows = {}
    for name, sign, initial in [('pole', 1, np.sqrt(1-.8**2)),
                                ('inner', -1, 1j*np.sqrt(1-.8**2)),
                                ('outer', -1, np.sqrt(1+.8**2))]:
        # Full quartic F, not just the reduced square-root equation.
        f = lambda q, a, s=sign: q**4+s*2*a*a*q*q+a**4-1
        fq = lambda q, a, s=sign: 4*q**3+s*4*a*a*q
        fa = lambda q, a, s=sign: s*4*a*q*q+4*a**3
        result = continue_root(path, np.array([initial]), f, fq, fa)
        row = summary(result)
        row['min_distance_to_discriminant'] = float(np.abs(result['parameters'][:, None]-np.array([1, -1, 1j, -1j])).min())
        row['returns_real'] = row['endpoint_imag_max'] < 1e-12
        rows[name] = row; results[name] = result
    real = []
    for eps in np.logspace(-2, -12, 11):
        a = 1-eps; z = np.sqrt(1-a*a)
        real.append(dict(a=float(a), pole=float(z), gradient=float(4*z), dz_da=float(a/z)))
    return dict(branches=rows, real_path=real,
                critical_origin=dict(a=1., F=0., gradient=0.),
                regular_complex_paths=all(x['max_residual'] < 1e-12 and x['min_spatial_jacobian'] > 0 for x in rows.values()),
                all_tracked_branches_return_real=all(x['returns_real'] for x in rows.values()),
                caveat='Pole starts real and ends imaginary; inner starts imaginary and ends real. Outer is real at both endpoints. Regular complex continuation does NOT guarantee real re-entry of every root.'), results


def local_equation(points, c, radius=.8):
    """Analytic continuation of the local real Wendland branch, within support.

    sqrt(sum x²), NOT Hermitian norm. Only used on tracked paths inside the
    support and away from its radial square-root branch cut. This does not
    assert a global holomorphic compactly supported field.
    """
    r = np.sqrt(np.sum(points*points, axis=-1))
    q = r/radius
    b = (1-q)**4*(4*q+1)
    s = np.sum(points*points, axis=-1)
    f = (s+1)**2-4*(points[..., 0]**2+points[..., 1]**2)-1+c*b
    g = 4*(s+1)[..., None]*points
    g[..., :2] -= 8*points[..., :2]
    g += c*(-20*(1-q)**3/radius**2)[..., None]*points
    return f, g, b
