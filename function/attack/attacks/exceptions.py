from typing import List, Tuple, Type, Union


class EstimatorError(TypeError):
    """
    Basic exception for errors raised by unexpected estimator types.
    """

    def __init__(self, this_class, class_expected_list: List[Union[Type, Tuple[Type]]], classifier_given) -> None:
        super().__init__()
        self.this_class = this_class
        self.class_expected_list = class_expected_list
        self.classifier_given = classifier_given

        classes_expected_message = ""
        for idx, class_expected in enumerate(class_expected_list):
            if idx != 0:
                classes_expected_message += " and "
            if isinstance(class_expected, type):
                classes_expected_message += f"{class_expected}"
            else:
                classes_expected_message += "("
                for or_idx, or_class in enumerate(class_expected):
                    if or_idx != 0:
                        classes_expected_message += " or "
                    classes_expected_message += f"{or_class}"
                classes_expected_message += ")"

        self.message = (
            f"{this_class.__name__} requires an estimator derived from {classes_expected_message}, "
            f"the provided classifier is an instance of {type(classifier_given)} "
            f"and is derived from {classifier_given.__class__.__bases__}."
        )

    def __str__(self) -> str:
        return self.message
