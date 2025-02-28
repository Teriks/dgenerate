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

import dgenerate.types

from .arguments import (
    parse_args,
    ImageProcessHelpException,
    ImageProcessArgs,
    ImageProcessUsageError,
)

from .invoker import (
    invoke_image_process,
    invoke_image_process_events,
    InvokeImageProcessEvent,
    InvokeImageProcessEventStream,
)

# noinspection PyUnresolvedReferences
from .renderloop import (
    ImageProcessRenderLoop,
    RenderLoopEventStream,
    RenderLoopEvent,
    ImageGeneratedEvent,
    StartingAnimationEvent,
    StartingAnimationFileEvent,
    AnimationFileFinishedEvent,
    ImageFileSavedEvent,
    AnimationFinishedEvent,
    AnimationETAEvent,
    StartingGenerationStepEvent,
)

from .renderloopconfig import (
    ImageProcessRenderLoopConfigError,
    ImageProcessRenderLoopConfig,
)

__doc__ = """
Implements the behaviors of dgenerate's ``image-process`` sub-command and ``\\image_process`` config directive.
"""

__all__ = dgenerate.types.module_all()
