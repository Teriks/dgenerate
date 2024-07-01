import dgenerate.promptweighters.promptweighter as _promptweighter


class MyPromptWeighter(_promptweighter.PromptWeighter):
    """
    Demo prompt weighter extensibility
    """

    NAMES = ['my-weighter']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def translate_to_embeds(self,
                            pipeline,
                            device: str,
                            args: dict[str, any]):
        """
        Translate the pipeline prompt arguments to ``prompt_embeds`` and ``pooled_prompt_embeds`` as needed.

        :param pipeline: The pipeline object
        :param device: The device the pipeline modules are on
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

        # print the pipeline type enum, txt2img, img2img, or inpaint
        print(self.pipeline_type)

        # print the dgenerate dtype enum for the pipeline
        # you can convert this to its torch equivalent
        # dgenerate.pipelinewrapper.get_torch_dtype()
        print(self.dtype)

        # print the arguments that will be passed to the pipeline
        print(args)

        # do nothing to the arguments
        # effectively a no-op
        return args

    def cleanup(self):
        """
        Preform any cleanup required after translating the pipeline arguments to embeds
        """

        # if we created and large tensors or anything of the sort,
        # we would move them back to the CPU here and preform any
        # garbage collection that might be needed

        pass
