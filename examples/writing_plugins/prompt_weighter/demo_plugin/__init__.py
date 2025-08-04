import typing

import dgenerate.promptweighters.promptweighter as _promptweighter


class MyPromptWeighter(_promptweighter.PromptWeighter):
    """
    Demo prompt weighter extensibility
    """

    NAMES = ['my-weighter']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def translate_to_embeds(self,
                            pipeline,
                            args: dict[str, typing.Any]):
        """
        Translate the pipeline prompt arguments to ``prompt_embeds`` and ``pooled_prompt_embeds`` as needed.

        :param pipeline: The pipeline object
        :param args: Call arguments to the pipeline
        :return: ``args``, supplemented with prompt embedding arguments
        """

        # this function would normally remove any prompt, negative_prompt
        # arguments, or prompt_2 / negative_prompt_2 etc.
        # and replace them with embeds, leaving all other arguments
        # to the pipeline intact, unless the weighting implementation needs
        # to change other arguments as well.

        # print --model-type enum
        print(self.model_type)

        # print the dgenerate dtype enum for the pipeline
        # you can convert this to its torch equivalent
        # dgenerate.pipelinewrapper.get_torch_dtype()
        print(self.dtype)

        # The device to work on
        print(self.device)

        # print the arguments that will be passed to the pipeline
        print(args)

        # do nothing to the arguments
        # effectively a no-op
        return args

    def cleanup(self):
        """
        Perform any cleanup required after translating the pipeline arguments to embeds
        """

        # if we created and large tensors or anything of the sort,
        # we would move them back to the CPU here and perform any
        # garbage collection that might be needed

        pass
