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


import transformers

if not hasattr(transformers.DynamicCache, 'get_max_length'):
    # they changed this, get_max_length no longer exists
    # this breaks some LLMs with custom modeling code,
    # and who knows when those repositories will update.
    def get_max_length(self):
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        if hasattr(self, 'get_max_cache_shape'):
            # Use the new API if available
            max_shape = self.get_max_cache_shape()
            # DynamicCache returns -1 from get_max_cache_shape but should return None for get_max_length
            # to maintain backward compatibility (None = unlimited, -1 could be misinterpreted)
            if max_shape == -1 and isinstance(self, transformers.DynamicCache):
                return None
            return max_shape
        else:
            # Fallback for very old versions
            return None
    
    transformers.DynamicCache.get_max_length = get_max_length

if not hasattr(transformers.DynamicCache, 'seen_tokens'):
    # In newer versions of transformers, seen_tokens attribute was removed
    # and replaced with get_seq_length() method. This breaks custom modeling
    # code like Phi-3 that still expects the seen_tokens attribute.
    @property
    def seen_tokens(self):
        return self.get_seq_length()
    
    transformers.DynamicCache.seen_tokens = seen_tokens

if not hasattr(transformers.DynamicCache, 'get_usable_length'):
    # In newer versions of transformers, get_usable_length method was removed
    # from DynamicCache. This breaks custom modeling code like Phi-3 that
    # still expects this method to be available.
    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length() if hasattr(self, 'get_max_length') else None
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length
    
    transformers.DynamicCache.get_usable_length = get_usable_length