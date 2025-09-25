class ModelOutputs:
    def __init__(self, features=None, logits=None, **kwargs):
        self.dict = {'features': features, 'logits': logits}
        self.dict.update(kwargs)
    
    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value
    
    def __str__(self):
        return str(self.dict)

    def __repr__(self):
        return str(self.dict)

    def __getattr__(self, key):
        try:
            return self.dict[key]
        except KeyError as e:
            raise AttributeError(f"'ModelOutputs' object has no attribute '{key}'") from e

    def __contains__(self, key):
        return key in self.dict


def CreateModel(args):
    if args.method.lower() == 'deephit':
        from .DeepHit import DeepHit
        return DeepHit(args)
    elif args.method.lower() == 'ordsurv':
        from .OrdSurv import OrdSurv
        return OrdSurv(args)
    elif args.method.lower() == 'deepsurv':
        from .DeepSurv import DeepSurv
        return DeepSurv(args)
    elif args.method.lower() == 'nll':
        from .NLLSurv import NllSurv
        return NllSurv(args)
    elif args.method.lower() == 'angularord':
        from .AngularOrd import AngularOrdRep
        return AngularOrdRep(args)
    elif args.method.lower() == 'ordsoftmax':
        from .OrdSoftmax import OrdSoftmax
        return OrdSoftmax(args)
    elif args.method.lower() == 'vmf':
        from .vmf import VmfOrdinalModel
        return VmfOrdinalModel(args)
    elif args.method.lower() == 'softmax':
        from .Softmax import SoftmaxClassifier
        return SoftmaxClassifier(args)
    elif args.method.lower() == 'ordcls':
        from .OrdCls import OrdCls
        return OrdCls(args)
    elif args.method.lower() == 'lassocox':
        from .LassoCox import LassoCox
        return LassoCox(args)
    elif args.method.lower() == 'coxtime':
        from .CoxTime import CoxTime
        return CoxTime(args)
    elif args.method.lower() == 'decouple':
        from .decouple import Decouple
        return Decouple(args)
    else:
        raise ValueError(f"Unknown method: {args.method}.")