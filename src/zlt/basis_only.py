"""Single-scale analytic zero-set expansion with a fixed positive background.

No geometric base is stored or evaluated. Geometry exists only in coefficients.
"""
import torch
from .locality import BasisLayout,UniformGridIndex,wendland_values,wendland_gradients
from .matched_operator import basis_hessian
from .transverse_packet import FusedGridLookup


class BasisOnlyZeroSetField(FusedGridLookup):
    """Uses the existing fused-value/gradient transport interface, no grid.

    Support membership is independent of coefficient value, so derivatives at
    zero coefficients remain valid. Centers and radius are fixed metadata.
    """
    def __init__(self,layout:BasisLayout,coefficients,background=1.0):
        if len(coefficients)!=layout.count: raise ValueError('coefficient shape mismatch')
        if not torch.all(layout.radii==layout.radii[0]): raise ValueError('single scale required')
        if background!=1.0: raise ValueError('primary fixed background must be 1')
        self.layout=layout; self.coefficients=coefficients; self.background=1.0
        self.lower=layout.centers.new_full((3,),-1.2); self.upper=-self.lower
        self.index=UniformGridIndex(layout.centers,float(layout.radii[0]))
        self.support_lower=(layout.centers-layout.radii[:,None]).amin(0)
        self.support_upper=(layout.centers+layout.radii[:,None]).amax(0)

    def with_coefficients(self,coefficients):
        if coefficients.shape!=self.coefficients.shape: raise ValueError('fixed K required')
        result=object.__new__(type(self)); result.__dict__=dict(self.__dict__)
        result.coefficients=coefficients; return result

    def pairs(self,p):
        owners=((p>=self.support_lower)&(p<=self.support_upper)).all(1).nonzero().flatten()
        if not len(owners): return owners,owners
        s=self.index.query(p[owners],float(self.layout.radii[0]))
        return owners[s.point_ids],s.basis_ids

    def evaluate(self,p):
        pi,bi=self.pairs(p); d=p[pi]-self.layout.centers[bi]; r=self.layout.radii[bi]
        out=p.new_zeros((len(p),4)); out[:,0]=self.background
        terms=torch.cat((wendland_values(d,r)[:,None],wendland_gradients(d,r)),1)
        return out.index_add(0,pi,terms*self.coefficients[bi,None])

    def value(self,p): return self.evaluate(p)[:,0]
    def gradient(self,p): return self.evaluate(p)[:,1:]
    def jet(self,p):
        pi,bi=self.pairs(p); d=p[pi]-self.layout.centers[bi]; r=self.layout.radii[bi]
        gradient=p.new_zeros((len(p),3)).index_add(0,pi,wendland_gradients(d,r)*self.coefficients[bi,None])
        hessian=p.new_zeros((len(p),3,3)).index_add(0,pi,basis_hessian(d,r)*self.coefficients[bi,None,None])
        return gradient,hessian

    def state_dict(self):
        return dict(centers=self.layout.centers.detach().cpu(),radii=self.layout.radii.detach().cpu(),
            coefficients=self.coefficients.detach().cpu(),background=self.background,version='v091_basis_only_single_scale')

    @classmethod
    def from_state_dict(cls,state,device='cpu'):
        if state['version']!='v091_basis_only_single_scale': raise ValueError('wrong field version')
        return cls(BasisLayout(state['centers'].to(device),state['radii'].to(device)),state['coefficients'].to(device),state['background'])
