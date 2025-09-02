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

"""
Cross-platform font selection utilities.
"""

import platform
import sys
import tkinter as tk
import tkinter.font as tkfont


def get_monospace_font_families() -> list[str]:
    """
    Get a list of preferred monospace font families in order of preference.
    
    Returns platform-specific fonts that are known to render well and avoid
    bitmap font fallbacks that can appear grainy or aliased.
    
    :return: List of font family names in order of preference
    """
    system = platform.system()
    
    if system == 'Windows':
        # Windows fonts - prioritize modern, well-hinted fonts
        return [
            'Consolas',           # Modern, excellent hinting
            'Cascadia Code',      # Microsoft's new monospace font
            'Cascadia Mono',      # Variant without ligatures
            'JetBrains Mono',     # Popular programming font
            'Source Code Pro',    # Adobe's programming font
            'DejaVu Sans Mono',   # Good cross-platform fallback
            'Liberation Mono',    # Red Hat's font
            'Courier New',        # System fallback
            'Lucida Console',     # Windows system font
            'Monaco',             # Sometimes available
        ]
    elif system == 'Darwin':  # macOS
        # macOS fonts - system fonts render very well
        return [
            'SF Mono',            # Apple's system monospace font
            'Menlo',              # Traditional macOS monospace
            'Monaco',             # Classic Mac font
            'JetBrains Mono',     # Popular programming font
            'Source Code Pro',    # Adobe's programming font
            'Cascadia Code',      # Microsoft font, sometimes installed
            'DejaVu Sans Mono',   # Cross-platform fallback
            'Liberation Mono',    # Open source fallback
            'Courier New',        # System fallback
        ]
    else:  # Linux and other Unix-like systems
        # Linux fonts - prioritize fonts with good hinting
        return [
            'JetBrains Mono',     # Very popular on Linux
            'Source Code Pro',    # Adobe font, often installed
            'DejaVu Sans Mono',   # Excellent Linux font with good hinting
            'Liberation Mono',    # Red Hat's font, widely available
            'Ubuntu Mono',        # Ubuntu's system font
            'Droid Sans Mono',    # Google's font
            'Noto Sans Mono',     # Google's comprehensive font
            'Hack',               # Programming font
            'Fira Code',          # Mozilla's programming font
            'Cascadia Code',      # Microsoft font, sometimes available
            'Consolas',           # Sometimes available via Wine/packages
            'Inconsolata',        # Popular programming font
            'Anonymous Pro',      # Free programming font
            'Courier New',        # System fallback
            'Courier',            # Basic fallback
            'monospace',          # Generic fallback
        ]


def find_available_font(font_families: list[str], size: int = 10) -> tuple[str, int]:
    """
    Find the first available font from a list of font families.
    
    This function tests each font family to ensure it's actually available
    and functional on the current system.
    
    :param font_families: List of font family names to try in order
    :param size: Font size to use
    :return: Tuple of (font_family, size) for the first available font
    """
    # Get list of available font families
    try:
        available_families = set(tkfont.families())
    except tk.TclError:
        # Fallback if we can't get font families
        return ('Courier', size)
    
    # Try each font family in order of preference
    for family in font_families:
        if family in available_families:
            # Test that the font actually works by creating a Font object
            try:
                test_font = tkfont.Font(family=family, size=size)
                # If we can measure text with it, it's working
                test_font.measure("test")
                return (family, size)
            except (tk.TclError, Exception):
                # Font didn't work, try the next one
                continue
    
    # If no preferred fonts are available, fall back to system defaults
    system = platform.system()
    if system == 'Windows':
        return ('Courier New', size)
    elif system == 'Darwin':
        return ('Menlo', size)
    else:
        return ('monospace', size)


def get_default_monospace_font(size: int = 10) -> tuple[str, int]:
    """
    Get the best available monospace font for the current platform.
    
    This function automatically selects platform-appropriate monospace fonts
    that render cleanly without bitmap font fallbacks.
    
    :param size: Font size to use (default: 10)
    :return: Tuple of (font_family, size) for the best available monospace font
    """
    preferred_fonts = get_monospace_font_families()
    return find_available_font(preferred_fonts, size)


def create_monospace_font(size: int = 10, **kwargs) -> tkfont.Font:
    """
    Create a tkinter Font object with the best available monospace font.
    
    This creates a ready-to-use Font object that can be assigned to tkinter
    widgets to ensure consistent, high-quality monospace font rendering.
    
    :param size: Font size to use (default: 10)
    :param kwargs: Additional arguments to pass to :py:class:`tkinter.font.Font`
    :return: tkinter.font.Font object configured with the best monospace font
    """
    family, size = get_default_monospace_font(size)
    
    # Set default font properties for better rendering
    font_kwargs = {
        'family': family,
        'size': size,
    }
    
    # Update with any user-provided kwargs
    font_kwargs.update(kwargs)
    
    return tkfont.Font(**font_kwargs)


def get_default_ui_font(size: int = 9) -> tuple[str, int]:
    """
    Get the best available UI font for the current platform.
    
    This function selects platform-appropriate UI fonts that provide
    excellent readability for interface elements.
    
    :param size: Font size to use (default: 9)
    :return: Tuple of (font_family, size) for the best available UI font
    """
    system = platform.system()
    
    if system == 'Windows':
        # Windows UI fonts
        ui_fonts = [
            'Segoe UI',           # Modern Windows UI font
            'Tahoma',             # Older Windows UI font
            'Arial',              # Fallback
            'Helvetica',          # Cross-platform fallback
            'sans-serif',         # Generic fallback
        ]
    elif system == 'Darwin':  # macOS
        # macOS UI fonts
        ui_fonts = [
            'SF Pro Display',     # Modern macOS UI font
            'Helvetica Neue',     # Traditional macOS UI font
            'Helvetica',          # Classic fallback
            'Arial',              # System fallback
            'sans-serif',         # Generic fallback
        ]
    else:  # Linux and other Unix-like systems
        # Linux UI fonts
        ui_fonts = [
            'Ubuntu',             # Ubuntu's UI font
            'DejaVu Sans',        # Excellent Linux font
            'Liberation Sans',    # Red Hat's font
            'Noto Sans',          # Google's comprehensive font
            'Cantarell',          # GNOME's font
            'Open Sans',          # Popular web font
            'Arial',              # Common fallback
            'Helvetica',          # Classic fallback
            'sans-serif',         # Generic fallback
        ]
    
    return find_available_font(ui_fonts, size)


def create_ui_font(size: int = 9, **kwargs) -> tkfont.Font:
    """
    Create a tkinter Font object with the best available UI font.
    
    This creates a ready-to-use Font object optimized for user interface
    elements like labels, buttons, and menus.
    
    :param size: Font size to use (default: 9)
    :param kwargs: Additional arguments to pass to :py:class:`tkinter.font.Font`
    :return: tkinter.font.Font object configured with the best UI font
    """
    family, size = get_default_ui_font(size)
    
    font_kwargs = {
        'family': family,
        'size': size,
    }
    
    # Update with any user-provided kwargs
    font_kwargs.update(kwargs)
    
    return tkfont.Font(**font_kwargs)


def set_tkinter_font_defaults():
    """
    Set global tkinter font defaults to prevent bitmap font fallbacks.
    
    This function configures all the standard tkinter named fonts (TkDefaultFont,
    TkTextFont, TkFixedFont, etc.) to use high-quality vector fonts appropriate
    for the current platform. This ensures that all tkinter widgets will use
    clean, anti-aliased fonts instead of falling back to bitmap fonts.
    
    This should be called early in the application lifecycle, after tkinter
    is initialized but before creating widgets.
    
    :raises: Prints warnings to stderr if font configuration fails
    """
    try:
        # Get the root window to access font configuration
        root = tk._default_root
        if root is None:
            # Create a temporary root if none exists
            root = tk.Tk()
            root.withdraw()
            temp_root = True
        else:
            temp_root = False
            
        # Get platform-appropriate fonts
        ui_family, ui_size = get_default_ui_font(9)
        mono_family, mono_size = get_default_monospace_font(10)
        
        # Configure the default fonts
        # These are the standard tkinter font names that widgets use
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family=ui_family, size=ui_size)
        
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(family=ui_family, size=ui_size)
        
        fixed_font = tkfont.nametofont("TkFixedFont")
        fixed_font.configure(family=mono_family, size=mono_size)
        
        menu_font = tkfont.nametofont("TkMenuFont")
        menu_font.configure(family=ui_family, size=ui_size)
        
        heading_font = tkfont.nametofont("TkHeadingFont")
        heading_font.configure(family=ui_family, size=ui_size, weight="bold")
        
        caption_font = tkfont.nametofont("TkCaptionFont")
        caption_font.configure(family=ui_family, size=max(ui_size - 1, 8))
        
        small_caption_font = tkfont.nametofont("TkSmallCaptionFont")
        small_caption_font.configure(family=ui_family, size=max(ui_size - 2, 7))
        
        icon_font = tkfont.nametofont("TkIconFont")
        icon_font.configure(family=ui_family, size=ui_size)
        
        tooltip_font = tkfont.nametofont("TkTooltipFont")
        tooltip_font.configure(family=ui_family, size=max(ui_size - 1, 8))
        
        # Clean up temporary root if we created one
        if temp_root:
            root.destroy()
            
    except (tk.TclError, AttributeError, KeyError) as e:
        # If we can't set the defaults for any reason, don't crash
        # This might happen in headless environments or with unusual tkinter setups
        print(f"Warning: Could not set tkinter font defaults: {e}", file=sys.stderr)