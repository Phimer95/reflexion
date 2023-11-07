from typing import List, Optional, Union
from abc import abstractmethod, ABC

from generators.model import ModelBase


class Generator:
    @abstractmethod
    def self_reflection(self, func: str, feedback: str, model: ModelBase) -> str:
        ...

    @abstractmethod
    def func_impl(
        self,
        func_sig: str,
        model: ModelBase,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.0,
    ) -> Union[str, List[str]]:
        ...

    @abstractmethod
    def internal_tests(
            self,
            func_sig: str,
            model: ModelBase,
            max_num_tests: int = 5,
            temperature: float = 0.0,
    ) -> List[str]:
        ...
        
    @abstractmethod
    def generate_refined_test(
            self,
            func_sig: str,
            flawed_test: str,
            feedback: str,
            model: ModelBase,
    ) -> List[str]:
        ...
