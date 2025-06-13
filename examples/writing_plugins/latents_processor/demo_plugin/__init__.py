import typing

import diffusers
import torch

import dgenerate.latentsprocessors


# these are nearly identical to image processors aside from the
# overridable method signature / name where the latents are processed

class FooLatentsProcessor(dgenerate.latentsprocessors.LatentsProcessor):
    """My --latents-processor-help documentation, arguments are described automatically"""

    # This static property defines what names this module can be invoked by
    NAMES = ['foo']

    # you can hide base class arguments or any argument from URI usage
    # if your processor does not support the argument
    HIDE_ARGS = ['device', 'model-offload']

    def __init__(self,
                 my_argument: str,
                 my_argument_2: bool = False,
                 my_argument_3: float = 1.0,
                 my_argument_4: typing.Optional[float] = None,
                 my_argument_5: str | int | None = None,
                 **kwargs):
        super().__init__(**kwargs)

        self._my_argument = my_argument
        self._my_argument_2 = my_argument_2
        self._my_argument_3 = my_argument_3
        self._my_argument_4 = my_argument_4
        self._my_argument_5 = my_argument_5

        # you can raise custom argument errors with self.argument_error

        # raise self.argument_error('My argument error message')

        # This should be set if you are loading
        # large models, for cache statistics

        # self.set_size_estimate(1024)

        # if we are using a large model of some sort
        # we can fence cpu side memory in order to
        # free up any objects dgenerate has cached
        # if we need to make space in RAM

        # self.memory_guard_device('cpu', 1024)

        # same with objects cached in VRAM, this can
        # accept an ordinal, i.e. you can specify
        # which GPU to fence

        # self.memory_guard_device('cuda', 1024)

        # load a large object into CPU side memory
        # with caching and auto memory fencing

        # self._big_model1 = self.load_object_cached(
        #     tag='unique_tag_in_this_constructor1',
        #     estimated_size=1024,
        #     method=lambda: load_my_object(),
        #     memory_guard_device='cpu'
        # )

        # load a large object into GPU side memory
        # with caching and auto memory fencing

        # self._big_model2 = self.load_object_cached(
        #     tag='unique_tag_in_this_constructor2',
        #     estimated_size=1024,
        #     method=lambda: load_my_object(),
        #     memory_guard_device='gpu'
        # )

        # register anything that needs to
        # move onto the device requested by the user
        # anything registered should have a .to(device)
        # argument accepting at least one argument,
        # a torch.device reference or device string

        # self.register_module(have_to)

    def impl_process(self, pipeline: diffusers.DiffusionPipeline, latents: torch.Tensor):
        # modify the latents here

        print("FOO:", self._my_argument, self._my_argument_2, self._my_argument_3)
        return latents
