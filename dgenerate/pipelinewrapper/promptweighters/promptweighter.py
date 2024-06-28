# Copyright (c) 2023, Teriks
#
# dgenerate is distributed under the following BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.promptweighters.exceptions as _exceptions
import dgenerate.plugin as _plugin


class PromptWeighter(_plugin.Plugin):
    """
    Abstract base class for prompt weighter implementations.
    """

    def __init__(self,
                 loaded_by_name: str,
                 model_type: _enums.ModelType,
                 pipeline_type: _enums.PipelineType,
                 **kwargs):
        super().__init__(loaded_by_name=loaded_by_name,
                         argument_error_type=_exceptions.PromptWeighterArgumentError,
                         **kwargs)

        self._model_type = model_type
        self._pipeline_type = pipeline_type

    @property
    def model_type(self) -> _enums.ModelType:
        return self._model_type

    @property
    def pipeline_type(self) -> _enums.PipelineType:
        return self._pipeline_type

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
        pass

    def cleanup(self):
        """
        Preform any cleanup required after translating the pipeline arguments to embeds
        """
        pass
