import dgenerate.promptupscalers.promptupscaler as _promptupscaler
import dgenerate.prompt as _prompt


class MyPromptUpscaler(_promptupscaler.PromptUpscaler):
    """
    Demo prompt upscaler extensibility
    """

    NAMES = ['my-upscaler']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # This should be set if you are loading
        # large models, for cache statistics

        # self.set_size_estimate(1024)

        # if we are using a large model of some sort
        # we can fence cpu side memory in order to
        # free up any objects dgenerate has cached
        # if we need to make space in RAM

        # self.memory_fence_device('cpu', 1024)

        # same with objects cached in VRAM, this can
        # accept an ordinal, i.e. you can specify
        # which GPU to fence

        # self.memory_fence_device('cuda', 1024)

        # load a large object into CPU side memory
        # with caching and auto memory fencing

        # self._big_model1 = self.load_object_cached(
        #     tag='unique_tag_in_this_constructor1',
        #     estimated_size=1024,
        #     method=lambda: load_my_object(),
        #     memory_fence_device='cpu'
        # )

        # load a large object into GPU side memory
        # with caching and auto memory fencing

        # self._big_model2 = self.load_object_cached(
        #     tag='unique_tag_in_this_constructor2',
        #     estimated_size=1024,
        #     method=lambda: load_my_object(),
        #     memory_fence_device='gpu'
        # )

    def upscale(self, prompt: _prompt.Prompt) -> _prompt.PromptOrPrompts:
        """
        Upscale the prompt, possibly return multiple prompt variations if you want
        """

        new_prompt = _prompt.Prompt.copy(prompt)

        if not new_prompt.negative:
            new_prompt.negative = 'very low quality image'

        return new_prompt
