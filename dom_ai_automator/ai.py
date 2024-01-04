from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Generic, Type, TypeVar, cast

from fastapi import HTTPException
from glob_utils import robust  # type: ignore
from openai import AsyncOpenAI
from openai.types.chat.completion_create_params import Function
from pydantic import BaseModel

T = TypeVar("T")


class AIFunctionResponse(BaseModel, Generic[T]):
    function: str
    data: T


class AIFunction(BaseModel, Generic[T], ABC):
    @classmethod
    def get_type(cls):
        return T

    @classmethod
    @lru_cache
    def definition(cls) -> Function:
        assert cls.__doc__ is not None, "OpenAIFunction must have a docstring"
        _schema = cls.schema()  # type: ignore
        _name = cls.__name__.lower()
        _description = cls.__doc__
        _parameters = cast(
            dict[str, object],
            (
                {
                    "type": "object",
                    "properties": {
                        k: v for k, v in _schema["properties"].items() if k != "self"
                    },
                    "required": _schema.get("required", []),
                }
            ),
        )
        return Function(name=_name, description=_description, parameters=_parameters)

    @property
    def name_(self) -> str:
        return self.__class__.__name__.lower()

    @abstractmethod
    async def run(self) -> T:
        raise NotImplementedError

    @robust
    async def __call__(self) -> AIFunctionResponse[T]:
        response = await self.run()
        return AIFunctionResponse[T](function=self.name_, data=response)


class AIModel(AsyncOpenAI):
    async def openai_vision(self, *, text: str, url: str) -> str:
        chat = self.chat.completions
        response = await chat.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=2048,
        )
        data = response.choices[0].message.content
        assert isinstance(data, str), "No response from OpenAI"
        return data

    async def openai_chat(self, *, text: str, context: str) -> str:
        chat = self.chat.completions
        response = await chat.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "user", "content": text},
                {"role": "system", "content": context},
            ],
            max_tokens=2048,
        )
        data = response.choices[0].message.content
        assert isinstance(data, str), "No response from OpenAI"
        return data

    async def openai_instruct(self, *, text: str) -> str:
        response = await self.completions.create(
            model="gpt-3.5-turbo-instruct", prompt=text
        )
        data = response.choices[0].text
        return data

    async def function_call(self, text: str) -> Any:
        response = await self.chat.completions.create(
            messages=[{"content": text, "role": "user"}],
            model="gpt-4-1106-preview",
            functions=[cls.definition() for cls in AIFunction.__subclasses__()],
        )
        if len(response.choices) == 0:
            raise HTTPException(status_code=500, detail="No response from OpenAI")
        for choice in response.choices:
            function_call = choice.message.function_call
            if function_call is None:
                model_output = choice.message.content
                if model_output is None:
                    raise HTTPException(
                        status_code=500, detail="No response from OpenAI"
                    )
                return model_output
            function_name = function_call.name
            arguments = function_call.arguments
            for func in AIFunction.__subclasses__():
                if func.__name__.lower() == function_name:
                    return await _function(func, arguments)
            raise HTTPException(
                status_code=500, detail=f"Unknown function {function_name}"
            )


async def _function(func: Type[AIFunction[T]], arguments: str) -> AIFunctionResponse[T]:
    instance = func.parse_raw(arguments)
    return await instance()
