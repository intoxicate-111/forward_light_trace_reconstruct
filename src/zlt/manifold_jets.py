"""Frozen-chart normal deformation jets; no change to the transport operator.

The reference is the current field, not a new permanent geometry base.  This
first-order chart phase uses F_theta = F_ref - |grad F_ref| h. Polynomial modes
are ONLY tangent-plane modes. A C2 normal collar supplies a compact off-surface
extension (unity for |w| <= r/2, zero for |w| >= r), avoiding opposite sheets.
"""
import torch
from .locality import UniformGridIndex, wendland_values
from .transverse_packet import FusedGridLookup


def tangent_frame(normals):
    n = torch.nn.functional.normalize(normals, dim=1)
    if bool((normals.norm(dim=1) < 1e-12).any()):
        raise ValueError('degenerate chart normal')
    axis = torch.nn.functional.one_hot(n.abs().argmin(1), 3).to(n)
    e1 = torch.nn.functional.normalize(torch.linalg.cross(axis, n), dim=1)
    e2 = torch.linalg.cross(n, e1)
    return torch.stack((e1, e2, n), dim=1)


class JetLayout:
    def __init__(self, centers, normals, radii, order):
        if order not in (0, 1, 2):
            raise ValueError('only p=0,1,2 supported')
        if len(centers) == 0 or radii.shape != (len(centers),) or bool((radii <= 0).any()):
            raise ValueError('positive nonempty chart radii required')
        self.centers, self.radii, self.order = centers, radii, order
        self.frames = tangent_frame(normals)
        self.modes = (1, 3, 6)[order]
        self.count = len(centers) * self.modes
        self.query_radius = float(radii.max()) * 2**.5
        self.index = UniformGridIndex(centers, self.query_radius)

    def query(self, p):
        """Sparse (point, scalar-mode, h_mode, grad h_mode), never N x K."""
        pairs = self.index.query(p, self.query_radius)
        pi, ki = pairs.point_ids, pairs.basis_ids
        d = p[pi] - self.centers[ki]
        xyz = torch.einsum('eab,eb->ea', self.frames[ki], d) / self.radii[ki, None]
        keep = (xyz[:, :2].square().sum(1) < 1) & (xyz[:, 2].abs() < 1)
        pi, ki, xyz = pi[keep], ki[keep], xyz[keep]
        u, v, w = xyz.unbind(1)
        planar = torch.stack((u, v, u*0), 1)
        radius = torch.ones_like(u)
        window = wendland_values(planar, radius)
        # Algebraically the existing Wendland gradient; cancellation avoids
        # an artificial 0/0 regularization at the chart origin when taking H.
        q = planar.norm(dim=1)
        dw = -20*(1-q).clamp_min(0)[:, None]**3*planar
        # Quintic C2 collar, flat in the central half of the normal tube.
        t = (2*w.abs()-1).clamp(0, 1)
        collar = 1-10*t**3+15*t**4-6*t**5
        dc = (-30*t**2+60*t**3-30*t**4)*2*w.sign()
        one, zero = torch.ones_like(u), torch.zeros_like(u)
        poly = torch.stack((one, u, v, u*u, u*v, v*v), 1)[:, :self.modes]
        du = torch.stack((zero, one, zero, 2*u, v, zero), 1)[:, :self.modes]
        dv = torch.stack((zero, zero, one, zero, u, 2*v), 1)[:, :self.modes]
        val = window[:, None]*collar[:, None]*poly
        grad_local = torch.stack((
            collar[:, None]*(dw[:, 0, None]*poly+window[:, None]*du),
            collar[:, None]*(dw[:, 1, None]*poly+window[:, None]*dv),
            window[:, None]*dc[:, None]*poly), 2)
        grad = torch.einsum('ema,eab->emb', grad_local, self.frames[ki])/self.radii[ki, None, None]
        mode = ki[:, None]*self.modes+torch.arange(self.modes, device=p.device)
        return pi.repeat_interleave(self.modes), mode.flatten(), val.flatten(), grad.reshape(-1, 3)


class JetSupports:
    """Explicit dF/dtheta and its spatial derivative for existing tangent code."""
    def __init__(self, reference, layout):
        self.reference, self.layout = reference, layout
        self.radii = layout.radii.repeat_interleave(layout.modes)

    def query(self, p):
        pi, bi, h, dh = self.layout.query(p)
        g, H = self.reference.jet(p)
        gamma = g.norm(dim=1)
        dgamma = torch.einsum('nab,na->nb', H, g)/gamma[:, None].clamp_min(1e-30)
        return pi, bi, -gamma[pi]*h, -gamma[pi, None]*dh-h[:, None]*dgamma[pi]


class ManifoldJetField(FusedGridLookup):
    def __init__(self, reference, layout, coefficients, enabled=True):
        if coefficients.shape != (layout.count,):
            raise ValueError('scalar coefficient count mismatch')
        self.reference, self.layout, self.coefficients = reference, layout, coefficients
        self.enabled = enabled
        self.lower, self.upper = reference.lower, reference.upper
        self.supports = JetSupports(reference, layout)

    def with_coefficients(self, coefficients):
        return type(self)(self.reference, self.layout, coefficients, self.enabled)

    def evaluate(self, p):
        original = self.reference.evaluate(p)
        if not self.enabled:
            return original
        pi, bi, b, db = self.supports.query(p)
        return original.index_add(0, pi, torch.cat((b[:, None], db), 1)*self.coefficients[bi, None])

    def value(self, p):
        return self.evaluate(p)[:, 0]

    def gradient(self, p):
        return self.evaluate(p)[:, 1:]

    def jet(self, p):
        if not self.enabled:
            return self.reference.jet(p)
        # Small diagnostic blocks only. AD differentiates the exact analytic
        # gradient, including spatial variation in |grad F_ref| (not SDF=1).
        preserve_graph = torch.is_grad_enabled() and p.requires_grad
        with torch.enable_grad():
            x = p if preserve_graph else p.detach().requires_grad_(True)
            g = self.gradient(x)
            H = torch.stack([torch.autograd.grad(g[:, a].sum(), x, retain_graph=preserve_graph or a < 2,
                                               create_graph=preserve_graph)[0]
                             for a in range(3)], 1)
        return (g, H) if preserve_graph else (g.detach(), H.detach())
