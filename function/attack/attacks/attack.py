from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import logging
from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from .summary_writer import SummaryWriter, SummaryWriterDefault
if TYPE_CHECKING:
    from function.attack.attacks.utils import CLASSIFIER_TYPE, GENERATOR_TYPE

class InputFilter(abc.ABCMeta):  # pragma: no cover
    def __init__(cls, name, bases, clsdict):  # pylint: disable=W0231

        def make_replacement(fdict, func_name):
            def replacement_function(self, *args, **kwargs):
                if len(args) > 0:
                    lst = list(args)

                if "x" in kwargs:
                    if not isinstance(kwargs["x"], np.ndarray):
                        kwargs["x"] = np.array(kwargs["x"])
                else:
                    if not isinstance(args[0], np.ndarray):
                        lst[0] = np.array(args[0])

                if "y" in kwargs:
                    if kwargs["y"] is not None and not isinstance(kwargs["y"], np.ndarray):
                        kwargs["y"] = np.array(kwargs["y"])
                elif len(args) == 2:
                    if not isinstance(args[1], np.ndarray):
                        lst[1] = np.array(args[1])

                if len(args) > 0:
                    args = tuple(lst)
                return fdict[func_name](self, *args, **kwargs)

            replacement_function.__doc__ = fdict[func_name].__doc__
            replacement_function.__name__ = "new_" + func_name
            return replacement_function

        replacement_list = ["generate", "extract"]
        for item in replacement_list:
            if item in clsdict:
                new_function = make_replacement(clsdict, item)
                setattr(cls, item, new_function)


class Attack(abc.ABC):
    attack_params: List[str] = []
    # The _estimator_requirements define the requirements an estimator must satisfy to be used as a target for an
    # attack. They should be a tuple of requirements, where each requirement is either a class the estimator must
    # inherit from, or a tuple of classes which define a union, i.e. the estimator must inherit from at least one class
    # in the requirement tuple.
    _estimator_requirements: Optional[Union[Tuple[Any, ...], Tuple[()]]] = None

    def __init__(
        self,
        estimator,
        summary_writer: Union[str, bool, SummaryWriter] = False,
    ):
        super().__init__()

        if self.estimator_requirements is None:
            raise ValueError("Estimator requirements have not been defined in `_estimator_requirements`.")

        # if not self.is_estimator_valid(estimator, self._estimator_requirements):
        #     raise EstimatorError(self.__class__, self.estimator_requirements, estimator)

        self._estimator = estimator
        self._summary_writer_arg = summary_writer
        self._summary_writer: Optional[SummaryWriter] = None

        # if isinstance(summary_writer, SummaryWriter):  # pragma: no cover
        #     self._summary_writer = summary_writer
        # elif summary_writer:
        #     self._summary_writer = SummaryWriterDefault(summary_writer)

        # Attack._check_params(self)

    @property
    def estimator(self):
        """The estimator."""
        return self._estimator

    @property
    def summary_writer(self):
        """The summary writer."""
        return self._summary_writer

    @property
    def estimator_requirements(self):
        """The estimator requirements."""
        return self._estimator_requirements

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        self._check_params()

    def _check_params(self) -> None:

        if not isinstance(self._summary_writer_arg, (bool, str, SummaryWriter)):
            raise ValueError("The argument `summary_writer` has to be either of type bool or str.")

    @staticmethod
    def is_estimator_valid(estimator, estimator_requirements) -> bool:
        for req in estimator_requirements:
            # A requirement is either a class which the estimator must inherit from, or a tuple of classes and the
            # estimator is required to inherit from at least one of the classes
            if isinstance(req, tuple):
                if all(p not in type(estimator).__mro__ for p in req):
                    return False
            elif req not in type(estimator).__mro__:
                return False
        return True


class EvasionAttack(Attack):
    def __init__(self, **kwargs) -> None:
        self._targeted = False
        super().__init__(**kwargs)

    @abc.abstractmethod
    def generate(  # lgtm [py/inheritance/incorrect-overridden-signature]
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        raise NotImplementedError

    @property
    def targeted(self) -> bool:
        return self._targeted

    @targeted.setter
    def targeted(self, targeted) -> None:
        self._targeted = targeted


class PoisoningAttack(Attack):
    def __init__(self, classifier: Optional["CLASSIFIER_TYPE"]) -> None:
        super().__init__(classifier)

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y=Optional[np.ndarray], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class PoisoningAttackGenerator(Attack):
    def __init__(self, generator: "GENERATOR_TYPE") -> None:
        super().__init__(generator)

    @abc.abstractmethod
    def poison_estimator(
        self,
        z_trigger: np.ndarray,
        x_target: np.ndarray,
        batch_size: int,
        max_iter: int,
        lambda_p: float,
        verbose: int,
        **kwargs
    ) -> "GENERATOR_TYPE":
        raise NotImplementedError

    @property
    def z_trigger(self):
        return self._z_trigger

    @property
    def x_target(self):
        return self._x_target


class PoisoningAttackTransformer(PoisoningAttack):
    def __init__(self, classifier: Optional["CLASSIFIER_TYPE"]) -> None:
        super().__init__(classifier)

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y=Optional[np.ndarray], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def poison_estimator(self, x: np.ndarray, y: np.ndarray, **kwargs) -> "CLASSIFIER_TYPE":
        raise NotImplementedError


class PoisoningAttackBlackBox(PoisoningAttack):
    def __init__(self):
        super().__init__(None)  # type: ignore

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class PoisoningAttackWhiteBox(PoisoningAttack):
    @abc.abstractmethod
    def poison(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class ExtractionAttack(Attack):
    @abc.abstractmethod
    def extract(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> "CLASSIFIER_TYPE":
        raise NotImplementedError


class InferenceAttack(Attack):

    def __init__(self, estimator):
        super().__init__(estimator)

    @abc.abstractmethod
    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        raise NotImplementedError


class AttributeInferenceAttack(InferenceAttack):
    attack_params = InferenceAttack.attack_params + ["attack_feature"]

    def __init__(self, estimator, attack_feature: Union[int, slice] = 0):
        super().__init__(estimator)
        self.attack_feature = attack_feature

    @abc.abstractmethod
    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        raise NotImplementedError


class MembershipInferenceAttack(InferenceAttack):
    def __init__(self, estimator):
        super().__init__(estimator)

    @abc.abstractmethod
    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        # Save attack-specific parameters
        super().set_params(**kwargs)
        self._check_params()


class ReconstructionAttack(Attack):
    attack_params = InferenceAttack.attack_params

    def __init__(self, estimator):
        super().__init__(estimator)

    @abc.abstractmethod
    def reconstruct(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        # Save attack-specific parameters
        super().set_params(**kwargs)
        self._check_params()