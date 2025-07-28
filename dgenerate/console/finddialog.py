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
import inspect
import re
import tkinter as tk
import typing
import dataclasses

import dgenerate.console.util as _util
import dgenerate.console.helpdialog as _helpdialog


@dataclasses.dataclass
class SearchOptions:
    """Configuration for search operations"""
    case_sensitive: bool = False
    regex_mode: bool = False
    wrap_around: bool = True


@dataclasses.dataclass
class SearchResult:
    """Result of a search operation"""
    found: bool
    start_pos: str = ""
    end_pos: str = ""
    match_text: str = ""


class MatchSpan(typing.NamedTuple):
    """Represents a match with start and end positions"""
    start: str
    end: str


class SearchEngine:
    """Searches the text widget"""
    
    def __init__(self, text_widget: tk.Text):
        self.text_widget = text_widget
        self._compiled_pattern = None
        self._last_pattern = ""
        self._last_options = None
    
    def _get_pattern(self, pattern: str, options: SearchOptions) -> typing.Optional[re.Pattern]:
        """Get compiled regex pattern with caching"""
        if (pattern == self._last_pattern and 
            options.case_sensitive == getattr(self._last_options, 'case_sensitive', None) and
            options.regex_mode == getattr(self._last_options, 'regex_mode', None)):
            return self._compiled_pattern
        
        try:
            if options.regex_mode:
                flags = 0 if options.case_sensitive else re.IGNORECASE
                self._compiled_pattern = re.compile(pattern, flags)
            else:
                escaped = re.escape(pattern)
                flags = 0 if options.case_sensitive else re.IGNORECASE
                self._compiled_pattern = re.compile(escaped, flags)
            
            self._last_pattern = pattern
            self._last_options = options
            return self._compiled_pattern
        except re.error:
            return None
    
    def find_next(self, pattern: str, start_pos: str, options: SearchOptions) -> SearchResult:
        """Find next occurrence of pattern"""
        if not pattern:
            return SearchResult(found=False)
        
        compiled_pattern = self._get_pattern(pattern, options)
        if not compiled_pattern:
            return SearchResult(found=False)
        
        if options.regex_mode:
            return self._find_next_regex(compiled_pattern, start_pos, options.wrap_around)
        else:
            return self._find_next_literal(pattern, start_pos, options)
    
    def find_previous(self, pattern: str, start_pos: str, options: SearchOptions) -> SearchResult:
        """Find previous occurrence of pattern"""
        if not pattern:
            return SearchResult(found=False)
        
        compiled_pattern = self._get_pattern(pattern, options)
        if not compiled_pattern:
            return SearchResult(found=False)
        
        return self._find_previous_regex(compiled_pattern, start_pos, options.wrap_around)
    
    def find_all(self, pattern: str, options: SearchOptions) -> typing.List[MatchSpan]:
        """Find all occurrences of pattern"""
        if not pattern:
            return []
        
        compiled_pattern = self._get_pattern(pattern, options)
        if not compiled_pattern:
            return []
        
        text = self.text_widget.get('1.0', tk.END)
        matches = []
        
        for match in compiled_pattern.finditer(text):
            start_idx = self.text_widget.index(f'1.0 + {match.start()} chars')
            end_idx = self.text_widget.index(f'1.0 + {match.end()} chars')
            matches.append(MatchSpan(start_idx, end_idx))
        
        return matches
    
    def _find_next_regex(self, pattern: re.Pattern, start_pos: str, wrap: bool) -> SearchResult:
        """Find next using regex"""
        text = self.text_widget.get('1.0', tk.END)
        start_offset = self._pos_to_offset(start_pos)
        
        match = pattern.search(text, start_offset)
        if match:
            start_idx = self.text_widget.index(f'1.0 + {match.start()} chars')
            end_idx = self.text_widget.index(f'1.0 + {match.end()} chars')
            return SearchResult(True, start_idx, end_idx, match.group())
        
        # Wrap around if enabled
        if wrap and start_offset > 0:
            match = pattern.search(text, 0)
            if match and match.start() < start_offset:
                start_idx = self.text_widget.index(f'1.0 + {match.start()} chars')
                end_idx = self.text_widget.index(f'1.0 + {match.end()} chars')
                return SearchResult(True, start_idx, end_idx, match.group())
        
        return SearchResult(found=False)
    
    def _find_next_literal(self, pattern: str, start_pos: str, options: SearchOptions) -> SearchResult:
        """Find next using literal search"""
        pos = self.text_widget.search(
            pattern, start_pos,
            nocase=not options.case_sensitive,
            stopindex=tk.END
        )
        
        if pos:
            end_pos = f'{pos}+{len(pattern)}c'
            return SearchResult(True, pos, end_pos, pattern)
        
        # Wrap around if enabled
        if options.wrap_around:
            pos = self.text_widget.search(
                pattern, '1.0',
                nocase=not options.case_sensitive,
                stopindex=start_pos
            )
            if pos:
                end_pos = f'{pos}+{len(pattern)}c'
                return SearchResult(True, pos, end_pos, pattern)
        
        return SearchResult(found=False)
    
    def _find_previous_regex(self, pattern: re.Pattern, start_pos: str, wrap: bool) -> SearchResult:
        """Find previous using regex"""
        end_offset = self._pos_to_offset(start_pos)
        text = self.text_widget.get('1.0', start_pos)
        
        matches = list(pattern.finditer(text))
        if matches:
            match = matches[-1]
            start_idx = self.text_widget.index(f'1.0 + {match.start()} chars')
            end_idx = self.text_widget.index(f'1.0 + {match.end()} chars')
            return SearchResult(True, start_idx, end_idx, match.group())
        
        # Wrap around if enabled
        if wrap:
            full_text = self.text_widget.get('1.0', tk.END)
            matches = list(pattern.finditer(full_text))
            # Find last match that comes after current position
            for match in reversed(matches):
                if match.start() >= end_offset:
                    start_idx = self.text_widget.index(f'1.0 + {match.start()} chars')
                    end_idx = self.text_widget.index(f'1.0 + {match.end()} chars')
                    return SearchResult(True, start_idx, end_idx, match.group())
        
        return SearchResult(found=False)
    
    def _pos_to_offset(self, pos: str) -> int:
        """Convert text widget position to character offset"""
        return len(self.text_widget.get('1.0', pos))
    
    def validate_pattern(self, pattern: str, regex_mode: bool) -> bool:
        """Validate if pattern is valid"""
        if not pattern:
            return True
        
        if not regex_mode:
            return True
        
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False


class ReplaceEngine:
    """Handles text replacement operations"""
    
    def __init__(self, text_widget: tk.Text, search_engine: SearchEngine):
        self.text_widget = text_widget
        self.search_engine = search_engine
    
    def replace_current(self, find_pattern: str, replace_text: str, 
                       current_match: SearchResult, options: SearchOptions) -> bool:
        """Replace the currently selected match"""
        if not current_match.found:
            return False
        
        # For regex, expand replacement groups
        if options.regex_mode:
            pattern = self.search_engine._get_pattern(find_pattern, options)
            if pattern:
                match_text = self.text_widget.get(current_match.start_pos, current_match.end_pos)
                match_obj = pattern.search(match_text)
                if match_obj:
                    replace_text = self._expand_replacement(match_obj, replace_text)
        
        self.text_widget.edit_separator()
        self.text_widget.replace(current_match.start_pos, current_match.end_pos, replace_text)
        return True
    
    def replace_all(self, find_pattern: str, replace_text: str, options: SearchOptions) -> int:
        """Replace all occurrences"""
        if not find_pattern:
            return 0
        
        pattern = self.search_engine._get_pattern(find_pattern, options)
        if not pattern:
            return 0
        
        text = self.text_widget.get('1.0', tk.END)
        
        self.text_widget.edit_separator()
        
        if options.regex_mode:
            try:
                # Always use custom expansion in regex mode to handle \0, \{0}, etc.
                new_text, count = pattern.subn(lambda m: self._expand_replacement(m, replace_text), text)
                self.text_widget.replace('1.0', tk.END, new_text)
                return count
            except re.error:
                return 0
        else:
            count = text.count(find_pattern)
            if count > 0:
                new_text = text.replace(find_pattern, replace_text)
                self.text_widget.replace('1.0', tk.END, new_text)
            return count
    
    def _expand_replacement(self, match: re.Match, replacement: str) -> str:
        """Expand regex replacement groups"""
        # First, let's handle the simple case where Python's expand() works
        if '\\0' not in replacement and '\\{' not in replacement:
            try:
                return match.expand(replacement)
            except re.error:
                pass
        
        # Manual expansion needed
        result = replacement
        
        # Get max group number
        max_group = match.lastindex if match.lastindex else 0
        
        # Create a mapping of all possible group references
        groups = {}
        for i in range(0, max_group + 1):
            try:
                groups[i] = match.group(i) or ""
            except IndexError:
                groups[i] = ""
        
        # Replace \{N} format first (more specific)
        def replace_brace_format(m):
            group_num = int(m.group(1))
            return groups.get(group_num, m.group(0))
        
        result = re.sub(r'\\{(\d+)}', replace_brace_format, result)
        
        # Replace \N format (less specific)
        def replace_backslash_format(m):
            group_num = int(m.group(1))
            return groups.get(group_num, m.group(0))
        
        result = re.sub(r'\\(\d+)', replace_backslash_format, result)
        
        return result
    
    def validate_replacement(self, find_pattern: str, replace_text: str, options: SearchOptions) -> bool:
        """Validate replacement pattern"""
        if not options.regex_mode or not replace_text:
            return True
        
        pattern = self.search_engine._get_pattern(find_pattern, options)
        if not pattern:
            return True  # Invalid find pattern, can't validate replacement
        
        # Test the replacement string by trying to use it with a dummy match
        try:
            # Create a dummy match to test replacement expansion
            dummy_text = "test"
            dummy_match = pattern.search(dummy_text * pattern.groups + dummy_text)
            if dummy_match:
                # Try to expand the replacement to catch syntax errors
                dummy_match.expand(replace_text)
            
            # Also check for invalid group references manually
            for match in re.finditer(r'\\(\d+)|\\{(\d+)}', replace_text):
                group_num = int(next(g for g in match.groups() if g))
                # \0 (entire match) is always valid, check others against available groups
                if group_num > 0 and group_num > pattern.groups:
                    return False
            
            return True
        except (re.error, IndexError):
            return False


class HighlightManager:
    """Manages text highlighting for search results"""
    
    def __init__(self, text_widget: tk.Text):
        self.text_widget = text_widget
        self._configure_tags()

        og_config = self.text_widget.config
        og_configure = self.text_widget.configure

        def new_config(*args, **kwargs):
            r = og_config(*args, **kwargs)
            self._configure_tags()
            return r

        def new_configure(*args, **kwargs):
            r = og_configure(*args, **kwargs)
            self._configure_tags()
            return r

        self.text_widget.config = new_config
        self.text_widget.configure = new_configure
    
    def _configure_tags(self):
        """Configure text tags for highlighting"""
        # Get the selection colors from the text widget
        try:
            sel_bg = self.text_widget.cget('selectbackground')
            sel_fg = self.text_widget.cget('selectforeground')
        except tk.TclError:
            # Fallback colors if we can't get them
            sel_bg = 'lightblue'
            sel_fg = 'black'
        
        # Use the same colors as selection for consistency
        self.text_widget.tag_config('found', 
                                    background=sel_bg,
                                    foreground=sel_fg)
        
        # Lower the priority of 'found' tag so actual selection takes precedence
        self.text_widget.tag_lower('found')
    
    def highlight_match(self, match: SearchResult):
        """Highlight a single match"""
        self.clear_highlights()
        if match.found:
            self.text_widget.tag_add('found', match.start_pos, match.end_pos)
            # Only ensure visibility, don't move cursor to preserve user's position
            self.text_widget.see(match.start_pos)
    
    def highlight_all_matches(self, matches: typing.List[MatchSpan]):
        """Highlight all matches"""
        self.clear_highlights()
        for match in matches:
            self.text_widget.tag_add('found', match.start, match.end)
    
    def clear_highlights(self):
        """Clear all search highlights"""
        self.text_widget.tag_remove('found', '1.0', tk.END)
    
    def mark_invalid(self, widget: tk.Widget, invalid: bool):
        """Mark input widget as invalid"""
        if hasattr(widget, 'tag_remove'):  # Text widget
            widget.tag_remove('invalid', '1.0', tk.END)
            if invalid:
                widget.tag_add('invalid', '1.0', tk.END)
            else:
                # Reset to normal appearance by ensuring no invalid tag
                try:
                    # Get the normal background color from the widget
                    normal_bg = widget.cget('bg')
                    if normal_bg in ('#ffcccc', 'red'):  # If it's still showing error color
                        widget.config(bg='white')
                except tk.TclError:
                    pass
        else:  # Entry widget
            if invalid:
                widget.config(bg='#ffcccc')
            else:
                widget.config(bg='white')


class DialogState:
    """Manages persistent dialog state"""
    
    def __init__(self):
        self.find_text = ""
        self.replace_text = ""
        self.case_sensitive = False
        self.regex_mode = False
        self.last_position = None
        self.last_find_width = None
        self.last_replace_width = None
        self.current_match = SearchResult(found=False)


class FindDialog(tk.Toplevel):
    """Modern, clean find/replace dialog implementation"""
    
    def __init__(self, master, name: str, text_widget: tk.Text,
                 replace_mode: bool = False, state: typing.Optional[DialogState] = None):
        super().__init__(master)
        
        self.master = master
        self.text_widget = text_widget
        self.replace_mode = replace_mode
        self.state = state or DialogState()
        
        # Initialize engines
        self.search_engine = SearchEngine(text_widget)
        self.replace_engine = ReplaceEngine(text_widget, self.search_engine)
        self.highlight_manager = HighlightManager(text_widget)
        
        # Event binding management
        self._event_bindings = []
        
        self._setup_ui(name)
        self._setup_events()
        self._restore_state()
        self._position_dialog()
    
    def _setup_ui(self, name: str):
        """Setup the user interface"""
        self.title(name)
        self.resizable(True, False)
        
        # Keep dialog visible on top while remaining modeless
        self.attributes('-topmost', True)
        
        # Find entry with frame for potential scrollbar
        self.find_frame = tk.Frame(self)
        self.find_frame.grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        self.find_frame.grid_columnconfigure(0, weight=1)
        
        self.find_entry = tk.Text(self.find_frame, height=1, width=40, wrap='none')
        self.find_entry.grid(row=0, column=0, sticky='ew')
        self.find_entry.tag_config('invalid', background='#ffcccc', foreground='#800000')
        
        # Create scrollbar but keep it hidden initially
        self.find_scrollbar = tk.Scrollbar(self.find_frame, orient='vertical', command=self.find_entry.yview)
        self.find_entry.configure(yscrollcommand=self.find_scrollbar.set)
        
        # Replace entry (if needed) with frame for potential scrollbar
        self.replace_entry = None
        self.replace_frame = None
        self.replace_scrollbar = None
        if self.replace_mode:
            self.replace_frame = tk.Frame(self)
            self.replace_frame.grid(row=1, column=0, sticky='ew', padx=2, pady=2)
            self.replace_frame.grid_columnconfigure(0, weight=1)
            
            self.replace_entry = tk.Text(self.replace_frame, height=1, width=40, wrap='none')
            self.replace_entry.grid(row=0, column=0, sticky='ew')
            self.replace_entry.tag_config('invalid', background='#ffcccc', foreground='#800000')
            
            # Create scrollbar but keep it hidden initially
            self.replace_scrollbar = tk.Scrollbar(self.replace_frame, orient='vertical', command=self.replace_entry.yview)
            self.replace_entry.configure(yscrollcommand=self.replace_scrollbar.set)
        
        # Options and buttons frame
        button_frame = tk.Frame(self)
        button_frame.grid(row=2, column=0, sticky='ew', padx=2, pady=2)
        
        # Options
        self.case_var = tk.BooleanVar(value=self.state.case_sensitive)
        self.regex_var = tk.BooleanVar(value=self.state.regex_mode)
        
        # Add callbacks to re-validate when options change
        self.case_var.trace_add('write', lambda *args: self._on_options_changed())
        self.regex_var.trace_add('write', lambda *args: self._on_options_changed())
        
        tk.Checkbutton(button_frame, text="Case Sensitive", 
                      variable=self.case_var).pack(side='left', padx=2)
        tk.Checkbutton(button_frame, text="Regex Search", 
                      variable=self.regex_var).pack(side='left', padx=2)
        
        # Navigation buttons
        tk.Button(button_frame, text='Previous', 
                 command=self._find_previous).pack(side='left', padx=2)
        tk.Button(button_frame, text='Next', 
                 command=self._find_next).pack(side='left', padx=2)
        
        # Replace buttons (if needed)
        if self.replace_mode:
            tk.Button(button_frame, text='Replace', 
                     command=self._replace_current).pack(side='left', padx=2)
        
        # Find All button
        tk.Button(button_frame, text='Find All', 
                 command=self._find_all).pack(side='left', padx=2)
        
        # Replace All button (if needed)
        if self.replace_mode:
            tk.Button(button_frame, text='Replace All', 
                     command=self._replace_all).pack(side='left', padx=2)
        
        # Help button (only in replace mode when regex is enabled)
        if self.replace_mode:
            self.help_button = tk.Button(button_frame, text='Help', 
                                       command=self._show_help)
            self.help_button.pack(side='right', padx=2)
            # Initially hide if regex mode is not enabled
            if not self.regex_var.get():
                self.help_button.pack_forget()
        
        self.grid_columnconfigure(0, weight=1)
    
    def _setup_events(self):
        """Setup event handlers"""
        # Find entry events
        self._bind_event(self.find_entry, '<KeyRelease>', self._on_find_text_changed)
        self._bind_event(self.find_entry, '<Return>', lambda e: (self._find_next(), "break")[1])
        
        # Replace entry events
        if self.replace_entry:
            self._bind_event(self.replace_entry, '<KeyRelease>', self._on_replace_text_changed)
            self._bind_event(self.replace_entry, '<Return>', lambda e: "break")
        
        # Text widget events to manage highlights
        self._bind_event(self.text_widget, '<Button-1>', self._on_text_click, add='+')
        self._bind_event(self.text_widget, '<B1-Motion>', self._on_text_select, add='+')
        
        # Window close event
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _bind_event(self, widget, event, handler, add=''):
        """Bind event and track for cleanup"""
        binding_id = widget.bind(event, handler, add)
        self._event_bindings.append((widget, event, binding_id))
        return binding_id
    
    def _get_search_options(self) -> SearchOptions:
        """Get current search options"""
        return SearchOptions(
            case_sensitive=self.case_var.get(),
            regex_mode=self.regex_var.get()
        )
    
    def _get_find_text(self) -> str:
        """Get current find text"""
        return self.find_entry.get('1.0', 'end-1c')
    
    def _get_replace_text(self) -> str:
        """Get current replace text"""
        if self.replace_entry:
            return self.replace_entry.get('1.0', 'end-1c')
        return ""
    
    def _find_next(self):
        """Find next occurrence"""
        pattern = self._get_find_text()
        if not pattern:
            return
        
        start_pos = self.state.current_match.end_pos if self.state.current_match.found else '1.0'
        if not start_pos or start_pos == '1.0':
            start_pos = self.text_widget.index(tk.INSERT)
        
        result = self.search_engine.find_next(pattern, start_pos, self._get_search_options())
        self.state.current_match = result
        self.highlight_manager.highlight_match(result)
    
    def _find_previous(self):
        """Find previous occurrence"""
        pattern = self._get_find_text()
        if not pattern:
            return
        
        start_pos = self.state.current_match.start_pos if self.state.current_match.found else tk.INSERT
        if not start_pos:
            start_pos = self.text_widget.index(tk.INSERT)
        
        result = self.search_engine.find_previous(pattern, start_pos, self._get_search_options())
        self.state.current_match = result
        self.highlight_manager.highlight_match(result)
    
    def _find_all(self):
        """Find and highlight all occurrences"""
        pattern = self._get_find_text()
        if not pattern:
            return
        
        matches = self.search_engine.find_all(pattern, self._get_search_options())
        self.highlight_manager.highlight_all_matches(matches)
        self.state.current_match = SearchResult(found=False)  # Clear current selection
    
    def _replace_current(self):
        """Replace current match"""
        if not self.state.current_match.found:
            return
        
        find_text = self._get_find_text()
        replace_text = self._get_replace_text()
        
        if self.replace_engine.replace_current(
            find_text, replace_text, self.state.current_match, self._get_search_options()
        ):
            # Find next after replacement
            self._find_next()
    
    def _replace_all(self):
        """Replace all occurrences"""
        find_text = self._get_find_text()
        replace_text = self._get_replace_text()
        
        if not find_text:
            return
        
        self.replace_engine.replace_all(find_text, replace_text, self._get_search_options())
        self.state.current_match = SearchResult(found=False)
        self.highlight_manager.clear_highlights()
    
    def _on_find_text_changed(self, event=None):
        """Handle find text changes"""
        pattern = self._get_find_text()
        options = self._get_search_options()
        
        # Check if we need to expand the entry box
        self._check_entry_expansion(self.find_entry, self.find_frame, 'find')
        
        # Validate pattern
        is_valid = self.search_engine.validate_pattern(pattern, options.regex_mode)
        self.highlight_manager.mark_invalid(self.find_entry, not is_valid)
        
        # Re-validate replacement when find pattern changes (affects group count)
        if self.replace_entry:
            self._on_replace_text_changed()
        
        # Clear current match when pattern changes
        self.state.current_match = SearchResult(found=False)
    
    def _on_replace_text_changed(self, event=None):
        """Handle replace text changes"""
        if not self.replace_entry:
            return

        # Check if we need to expand the entry box
        self._check_entry_expansion(self.replace_entry, self.replace_frame, 'replace')

        find_text = self._get_find_text()
        replace_text = self._get_replace_text()
        options = self._get_search_options()
        
        # Validate replacement
        is_valid = self.replace_engine.validate_replacement(find_text, replace_text, options)
        self.highlight_manager.mark_invalid(self.replace_entry, not is_valid)
    
    def _on_options_changed(self):
        """Handle when search options change (case sensitive, regex mode)"""
        # Re-validate both find and replace text when options change
        self._on_find_text_changed()
        if self.replace_entry:
            self._on_replace_text_changed()
        
        # Show/hide Help button based on regex mode
        if self.replace_mode and hasattr(self, 'help_button'):
            if self.regex_var.get():
                self.help_button.pack(side='right', padx=2)
            else:
                self.help_button.pack_forget()
    
    def _on_text_click(self, event):
        """Handle clicks in text widget"""
        # Don't clear highlights immediately - wait for selection
        # This allows clicking on highlighted text without losing the highlight
        pass
    
    def _on_text_select(self, event):
        """Handle text selection - clear highlights when user is selecting"""
        self.highlight_manager.clear_highlights()
        self.state.current_match = SearchResult(found=False)
    
    def _check_entry_expansion(self, entry_widget, frame_widget, entry_type):
        """Check if entry widget needs to expand and add scrollbar"""
        # Get the actual line count
        line_count = int(entry_widget.index('end-1c').split('.')[0])
        
        # Get scrollbar reference
        scrollbar_attr = f'{entry_type}_scrollbar'
        scrollbar = getattr(self, scrollbar_attr)
        
        # Track if size changed
        size_changed = False
        
        # If more than 1 line and not already expanded
        if line_count > 1 and entry_widget['height'] == 1:
            # Expand to 3 lines
            entry_widget.configure(height=3)
            size_changed = True
            
            # Show scrollbar
            if scrollbar:
                scrollbar.grid(row=0, column=1, sticky='nsew')
                
        # If back to 1 line and expanded
        elif line_count <= 1 and entry_widget['height'] > 1:
            # Shrink back to 1 line
            entry_widget.configure(height=1)
            size_changed = True
            
            # Hide scrollbar (but don't destroy it)
            if scrollbar:
                scrollbar.grid_forget()
        
        # If size changed, update the window geometry
        if size_changed:
            # Let Tkinter process the widget size changes
            self.update_idletasks()
            
            # Get the new required height
            req_height = self.winfo_reqheight()
            
            # Update window height while preserving width and position
            self.geometry(f"{self.winfo_width()}x{req_height}+{self.winfo_x()}+{self.winfo_y()}")
    
    def _show_help(self):
        """Show help dialog for substitution syntax"""
        help_text = \
        """When "Regex Search" is enabled, you can use special substitution 
           patterns in the replace field. This uses Python regex syntax (re module).
   
           Basic Substitutions:
           
           \\0     - Insert the entire match
           \\1     - Insert the 1st capture group  
           \\2     - Insert the 2nd capture group
           \\3     - Insert the 3rd capture group
           
           And so on...
   
           Alternative Syntax (for quoting next to digits):
           
           \\{0}   - Insert the entire match (same as \\0)
           \\{1}   - Insert the 1st capture group (same as \\1)
           \\{2}   - Insert the 2nd capture group (same as \\2)
           
           And so on...
           """

        _helpdialog.show_help_dialog(
            parent=self,
            title="Find & Replace - Substitution Syntax",
            help_text=inspect.cleandoc(help_text),
            size=(650, 500),
            dock_to_right=True,
            position_widget=self
        )
    
    def _restore_state(self):
        """Restore dialog state"""
        # Try to get selected text first
        try:
            selected = self.text_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.find_entry.insert('1.0', selected)
        except tk.TclError:
            if self.state.find_text:
                self.find_entry.insert('1.0', self.state.find_text)
        
        if self.replace_entry and self.state.replace_text:
            self.replace_entry.insert('1.0', self.state.replace_text)
        
        self.case_var.set(self.state.case_sensitive)
        self.regex_var.set(self.state.regex_mode)
        
        # Validate restored text to show any invalid patterns immediately
        self._on_find_text_changed()
        if self.replace_entry:
            self._on_replace_text_changed()
    
    def _position_dialog(self):
        """Position and size the dialog"""
        width = (self.state.last_replace_width if self.replace_mode 
                else self.state.last_find_width)
        
        self.withdraw()
        self.update_idletasks()
        
        if width:
            size = (width, self.winfo_reqheight())
        else:
            size = None
        
        _util.position_toplevel(
            master=self.master,
            toplevel=self,
            size=size,
            position=self.state.last_position
        )
        
        self.find_entry.focus_set()
    
    def _save_state(self):
        """Save current dialog state"""
        self.state.find_text = self._get_find_text()
        if self.replace_mode:
            self.state.replace_text = self._get_replace_text()
        
        self.state.case_sensitive = self.case_var.get()
        self.state.regex_mode = self.regex_var.get()
        self.state.last_position = (self.winfo_x(), self.winfo_y())
        
        if self.replace_mode:
            self.state.last_replace_width = self.winfo_width()
        else:
            self.state.last_find_width = self.winfo_width()
    
    def _on_closing(self):
        """Handle dialog closing"""
        self._save_state()
        self.destroy()
    
    def destroy(self):
        """Clean up and destroy dialog"""
        # Clean up event bindings
        for widget, event, binding_id in self._event_bindings:
            try:
                widget.unbind(event, binding_id)
            except tk.TclError:
                pass
        
        # Clear highlights
        try:
            self.highlight_manager.clear_highlights()
        except tk.TclError:
            pass
        
        super().destroy()


class FindReplaceDialogManager:
    """Manages find/replace dialog instances and state"""
    
    def __init__(self):
        self.state = DialogState()
        self.current_dialog = None
    
    def open_find_dialog(self, master, name: str, text_widget: tk.Text):
        """Open find dialog"""
        self._close_existing_dialog()
        
        self.current_dialog = FindDialog(
            master, name, text_widget, 
            replace_mode=False, state=self.state
        )
    
    def open_find_replace_dialog(self, master, name: str, text_widget: tk.Text):
        """Open find/replace dialog"""
        self._close_existing_dialog()
        
        self.current_dialog = FindDialog(
            master, name, text_widget, 
            replace_mode=True, state=self.state
        )
    
    def _close_existing_dialog(self):
        """Close existing dialog if open"""
        if (self.current_dialog and 
            hasattr(self.current_dialog, 'winfo_exists') and 
            self.current_dialog.winfo_exists()):
            
            # Check if we can reuse the dialog
            try:
                self.current_dialog.find_entry.focus_set()
                return
            except tk.TclError:
                pass
            
            self.current_dialog.destroy()
        
        self.current_dialog = None


# Global dialog manager instance
_dialog_manager = FindReplaceDialogManager()


def open_find_dialog(master, name: str, text_box: tk.Text):
    """Open find dialog"""
    _dialog_manager.open_find_dialog(master, name, text_box)


def open_find_replace_dialog(master, name: str, text_box: tk.Text):
    """Open find/replace dialog"""
    _dialog_manager.open_find_replace_dialog(master, name, text_box)
