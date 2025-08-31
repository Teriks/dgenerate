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
Main GUI application for the dgenerate network installer.
"""

import os
import platform
import shutil
import sys
import tempfile
import tempfile
import threading
import inspect
import tkinter as tk
import traceback
from datetime import datetime
from importlib.resources import files
from network_installer.github_client import GitHubClient
from network_installer.platform_detection import get_platform_info, detect_gpu, get_torch_index_url
from network_installer.setup_analyzer import SetupAnalyzer
from network_installer.uv_installer import UvInstaller
from tkinter import ttk, filedialog
from typing import List, Optional, Tuple


class DGenerateInstallerGUI:
    """
    Main GUI application for the dgenerate network installer.
    """

    def __init__(self):
        """
        Initialize the DGenerateInstallerGUI.
        """
        self.root = tk.Tk()
        self.root.title("dgenerate network installer")
        self.root.resizable(False, False)  # Fixed size like MSI installers

        # Center the window on screen like MSI installers
        window_width = 555  # 370 dialog units
        window_height = 405  # 270 dialog units

        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate center position
        center_x = (screen_width - window_width) // 2
        center_y = (screen_height - window_height) // 2

        # Set geometry with centered position
        self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

        # Set window icon
        icon_path, icon_format = self._load_icon()
        if icon_path and icon_format:
            try:
                if icon_format == 'ico':
                    self.root.iconbitmap(icon_path)
                elif icon_format == 'png':
                    # For PNG icons on Mac/Linux, we need to load the image
                    icon_image = tk.PhotoImage(file=icon_path)
                    self.root.iconphoto(True, icon_image)
                    # Keep a reference to prevent garbage collection
                    self.icon_image = icon_image
            except Exception as e:
                print(f"Warning: Could not set window icon: {e}")

        # Hook window close event to handle cancellation
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Store icon info for dialogs
        self.icon_path = icon_path
        self.icon_format = icon_format

        # Initialize global state variables
        self.github_client = GitHubClient()
        self.source_dir = None  # Global state - persists across all screens
        self.selected_extras = []
        self.available_extras = {}
        self.recommended_extras = []
        self.commit_hash = None
        self.branch_name = None
        self.is_pre_release = False

        # Global state for existing installation detection
        self.existing_install_data = None

        # Cache for releases and branches data
        self._releases_cache = None
        self._branches_cache = None

        # Create GUI
        self._create_widgets()
        self._setup_styles()

        # Check for existing installation first, before showing any screens
        self._check_existing_installation_first()

    def _show_progress_bar(self):
        """Show the progress bar during downloads."""
        self.progress_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        # Disable next button during download operations
        self.next_button.config(state="disabled")

    def _hide_progress_bar(self):
        """Hide the progress bar when not needed."""
        self.progress_bar.grid_remove()
        self.progress_var.set(0)
        # Re-enable next button when download completes
        self.next_button.config(state="normal")

    def _show_centered_messagebox(self, msg_type: str, title: str, message: str) -> bool:
        """
        Show a messagebox centered on the parent window.
        
        :param msg_type: Type of messagebox ('error', 'info', 'yesno')
        :param title: Title of the messagebox
        :param message: Message content
        :return: True/False for yesno, None for others
        """
        # Calculate center position first
        self.root.update_idletasks()
        parent_x = self.root.winfo_x()
        parent_y = self.root.winfo_y()
        parent_width = self.root.winfo_width()
        parent_height = self.root.winfo_height()

        # Set dialog size
        dialog_width = 350
        dialog_height = 150

        center_x = parent_x + (parent_width - dialog_width) // 2
        center_y = parent_y + (parent_height - dialog_height) // 2

        # Create dialog and set position immediately before showing
        dialog = tk.Toplevel(self.root)
        dialog.withdraw()  # Hide initially to prevent flicker
        dialog.title(title)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.geometry(f"{dialog_width}x{dialog_height}+{center_x}+{center_y}")

        # Set icon for dialog
        if hasattr(self, 'icon_path') and self.icon_path and hasattr(self, 'icon_format'):
            try:
                if self.icon_format == 'ico':
                    dialog.iconbitmap(self.icon_path)
                elif self.icon_format == 'png' and hasattr(self, 'icon_image'):
                    dialog.iconphoto(True, self.icon_image)
            except Exception:
                pass  # Ignore icon errors for dialogs

        # Make visible after positioning
        dialog.deiconify()
        dialog.grab_set()

        # Create dialog content
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Icon and message
        if msg_type == 'error':
            icon = "⚠️"
        elif msg_type == 'info':
            icon = "ℹ️"
        elif msg_type == 'yesno':
            icon = "❓"
        else:
            icon = ""

        icon_label = ttk.Label(main_frame, text=icon, font=("Arial", 16))
        icon_label.grid(row=0, column=0, padx=(0, 10), pady=(0, 10), sticky=tk.NW)

        message_label = ttk.Label(main_frame, text=message, wraplength=280, justify=tk.LEFT)
        message_label.grid(row=0, column=1, pady=(0, 20), sticky=tk.W)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))

        result = [None]  # Use list to store result for closure

        def on_ok():
            result[0] = True
            dialog.destroy()

        def on_cancel():
            result[0] = False
            dialog.destroy()

        if msg_type == 'yesno':
            yes_btn = ttk.Button(button_frame, text="Yes", command=on_ok)
            yes_btn.pack(side=tk.LEFT, padx=(0, 10))
            no_btn = ttk.Button(button_frame, text="No", command=on_cancel)
            no_btn.pack(side=tk.LEFT)
        else:
            ok_btn = ttk.Button(button_frame, text="OK", command=on_ok)
            ok_btn.pack()

        # Center the button frame
        button_frame.grid_configure(sticky="")

        # Wait for dialog to close
        dialog.wait_window()

        return result[0] if msg_type == 'yesno' else None

    def _load_icon(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Load icon file path from package resources.
        
        :return: Tuple of (path to icon file, icon format) or (None, None) if not found
                Format is either 'ico' for Windows or 'png' for Mac/Linux
        """
        try:
            # Determine the appropriate icon format based on platform
            system = platform.system().lower()
            if system == 'windows':
                icon_filename = 'icon.ico'
                icon_format = 'ico'
                suffix = '.ico'
            else:  # macOS and Linux
                icon_filename = 'icon.png'
                icon_format = 'png'
                suffix = '.png'

            # Try to load from package resources (works when installed as package)
            try:
                resource_files = files('network_installer.resources')
                icon_file = resource_files / icon_filename

                # For PyInstaller compatibility, we need to extract to a temp file
                temp_icon = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                temp_icon.write(icon_file.read_bytes())
                temp_icon.close()
                return temp_icon.name, icon_format
            except (ImportError, ModuleNotFoundError):
                # Fallback for PyInstaller - resources are bundled differently
                if hasattr(sys, '_MEIPASS'):
                    # PyInstaller bundle
                    icon_path = os.path.join(sys._MEIPASS, 'network_installer', 'resources', icon_filename)
                    if os.path.exists(icon_path):
                        return icon_path, icon_format
                else:
                    # Development mode - look for resources directory
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    icon_path = os.path.join(current_dir, 'resources', icon_filename)
                    if os.path.exists(icon_path):
                        return icon_path, icon_format

                return None, None

        except Exception as e:
            print(f"Warning: Could not load icon: {e}")
            return None, None

    def _load_license_text(self) -> str:
        """
        Load license text from package resources.
        
        :return: License text content
        """
        try:
            # Try to load from package resources (works when installed as package)
            try:
                resource_files = files('network_installer.resources')
                license_file = resource_files / 'LICENSE'
                return license_file.read_text(encoding='utf-8')
            except (ImportError, ModuleNotFoundError):
                # Fallback for PyInstaller - resources are bundled differently
                if hasattr(sys, '_MEIPASS'):
                    # PyInstaller bundle
                    license_path = os.path.join(sys._MEIPASS, 'network_installer', 'resources', 'LICENSE')
                    if os.path.exists(license_path):
                        with open(license_path, 'r', encoding='utf-8') as f:
                            return f.read()
                else:
                    # Development mode - look for resources directory
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    license_path = os.path.join(current_dir, 'resources', 'LICENSE')
                    if os.path.exists(license_path):
                        with open(license_path, 'r', encoding='utf-8') as f:
                            return f.read()

                # No fallback - return error for development visibility
                return f"License file not found in any expected location.\n\nRun build.py first to copy resources from main project."

        except Exception as e:
            # Show actual error for development debugging
            return f"Failed to load license text: {e}\n\nRun build.py first to copy resources from main project."

    def _create_widgets(self):
        """
        Create the main GUI widgets.
        """
        # Main frame with compact padding
        self.main_frame = ttk.Frame(self.root, padding="8")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

        # Remove title - it's already in window title bar

        # Content frame (will be replaced for different screens)
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_frame.rowconfigure(1, weight=1)
        self.content_frame.columnconfigure(0, weight=1)

        # Progress bar (initially hidden)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.main_frame, variable=self.progress_var,
                                            mode='determinate')
        # Don't grid it initially - will be shown only during downloads

        # Status label with compact spacing
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.main_frame, textvariable=self.status_var, font=("Arial", 8))
        self.status_label.grid(row=3, column=0, columnspan=2, pady=(3, 0))

        # Navigation buttons with compact spacing
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=4, column=0, columnspan=2, pady=(10, 0))

        self.back_button = ttk.Button(self.button_frame, text="Back", command=self._go_back)
        self.back_button.pack(side=tk.LEFT, padx=(0, 10))

        self.next_button = ttk.Button(self.button_frame, text="Next", command=self._go_next)
        self.next_button.pack(side=tk.LEFT)

        # Initialize navigation
        self.current_screen = 0
        self.screens = []

    def _setup_styles(self):
        """
        Set up custom styles for the GUI.
        """
        style = ttk.Style()

        # Configure styles
        style.configure("Title.TLabel", font=("Arial", 14, "bold"))
        style.configure("Subtitle.TLabel", font=("Arial", 12, "bold"))
        style.configure("Info.TLabel", font=("Arial", 10))

        # Configure button styles
        style.configure("Primary.TButton", font=("Arial", 10, "bold"))
        style.configure("Secondary.TButton", font=("Arial", 10))

    def _show_welcome_screen(self):
        """
        Show the welcome screen.
        """
        self._clear_content()

        # Welcome text - compact
        welcome_text = "This installer will download and install dgenerate with the appropriate components for your platform."

        welcome_label = ttk.Label(self.content_frame, text=welcome_text,
                                  wraplength=500, justify=tk.LEFT, font=("Arial", 9))
        welcome_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # License frame with compact padding
        license_frame = ttk.LabelFrame(self.content_frame, text="License Agreement", padding="5")
        license_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        license_frame.columnconfigure(0, weight=1)
        license_frame.rowconfigure(0, weight=1)

        # Load license text from package resources
        license_text = self._load_license_text()

        # Create scrollable text widget for license - larger and more compact font
        license_text_widget = tk.Text(license_frame, wrap=tk.WORD, height=20,
                                      font=("Consolas", 8), state=tk.DISABLED)
        license_text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Add scrollbar
        license_scrollbar = ttk.Scrollbar(license_frame, orient=tk.VERTICAL,
                                          command=license_text_widget.yview)
        license_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        license_text_widget.configure(yscrollcommand=license_scrollbar.set)

        # Insert license text
        license_text_widget.config(state=tk.NORMAL)
        license_text_widget.insert(tk.END, license_text)
        license_text_widget.config(state=tk.DISABLED)

        # System info (placed after license frame)
        self._show_system_info()

        # Configure content frame to expand properly
        self.content_frame.rowconfigure(1, weight=1)

        # Don't reset the screens array here - keep the current flow
        self.current_screen = 0
        self._update_navigation()

    def _show_system_info(self):
        """
        Show system information.
        """
        # Create system info frame with compact padding
        info_frame = ttk.LabelFrame(self.content_frame, text="System Information", padding="5")
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        info_frame.columnconfigure(0, weight=1)

        # Get system info
        platform_info = get_platform_info()
        gpu_info = detect_gpu()

        # Create info text
        info_text = f"""Platform: {platform_info.system.title()} {platform_info.architecture}"""

        if gpu_info.gpu_name:
            info_text += f"\nGPU: {gpu_info.gpu_name}"
            if gpu_info.cuda_version:
                info_text += f" (CUDA {gpu_info.cuda_version})"
            elif gpu_info.rocm_version:
                info_text += f" (ROCm {gpu_info.rocm_version})"

        info_label = ttk.Label(info_frame, text=info_text, font=("Arial", 8))
        info_label.grid(row=0, column=0, sticky=(tk.W, tk.E))

    def _show_source_selection_screen(self):
        """
        Show the source selection screen.
        """
        self._clear_content()

        # Title
        title_label = ttk.Label(self.content_frame, text="Select dgenerate source",
                                style="Title.TLabel")
        title_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        # Source options frame
        source_frame = ttk.LabelFrame(self.content_frame, text="Source Options", padding="10")
        source_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        source_frame.columnconfigure(0, weight=1)

        # Latest release option
        self.latest_release_var = tk.StringVar(value="latest")
        latest_radio = ttk.Radiobutton(source_frame, text="Latest Release",
                                       variable=self.latest_release_var, value="latest")
        latest_radio.grid(row=0, column=0, sticky=tk.W, pady=5)

        # Specific release option
        specific_radio = ttk.Radiobutton(source_frame, text="Specific Release",
                                         variable=self.latest_release_var, value="specific")
        specific_radio.grid(row=1, column=0, sticky=tk.W, pady=5)

        # Release selection combobox
        self.release_var = tk.StringVar()
        self.release_combo = ttk.Combobox(source_frame, textvariable=self.release_var,
                                          state="readonly", width=40)
        self.release_combo.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5, padx=(20, 0))

        # Add event handler for release combobox selection
        self.release_combo.bind('<<ComboboxSelected>>', self._on_release_selected)

        # Branch option
        branch_radio = ttk.Radiobutton(source_frame, text="Development Branch",
                                       variable=self.latest_release_var, value="branch")
        branch_radio.grid(row=3, column=0, sticky=tk.W, pady=5)

        # Branch selection combobox
        self.branch_var = tk.StringVar()
        self.branch_combo = ttk.Combobox(source_frame, textvariable=self.branch_var,
                                         state="readonly", width=40)
        self.branch_combo.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5, padx=(20, 0))

        # Add event handler for branch combobox selection
        self.branch_combo.bind('<<ComboboxSelected>>', self._on_branch_selected)

        # Load releases and branches
        self._load_releases_and_branches()

        # Don't reset the screens array here - keep the current flow
        self.current_screen = 1
        self._update_navigation()

    def _load_releases_and_branches(self):
        """
        Load releases and branches from GitHub.
        """
        # Check if we have cached data
        if self._releases_cache is not None and self._branches_cache is not None:
            # Use cached data
            self._update_source_lists(self._releases_cache, self._branches_cache)
            return

        def load_data():
            self.status_var.set("Loading releases and branches...")
            self._show_progress_bar()
            self.progress_var.set(0)

            # Load combined releases and tags (including pre-releases, increased limit)
            self.progress_var.set(25)
            self.status_var.set("Loading releases and tags...")
            releases = self.github_client.get_releases_and_tags_combined(per_page=500, include_prereleases=True)
            release_names = [f"{r['tag_name']} - {r['name']}" for r in releases]

            # Load branches
            self.progress_var.set(75)
            self.status_var.set("Loading branches...")
            branches = self.github_client.get_branches()
            branch_names = [b['name'] for b in branches]

            self.progress_var.set(100)

            # Cache the data
            self._releases_cache = release_names
            self._branches_cache = branch_names

            # Update GUI in main thread
            self.root.after(0, lambda: self._update_source_lists(release_names, branch_names))

        threading.Thread(target=load_data, daemon=True).start()

    def _update_source_lists(self, releases: List[str], branches: List[str]):
        """
        Update the release and branch lists.

        :param releases: List of release names
        :param branches: List of branch names
        """
        self.release_combo['values'] = releases
        if releases:
            self.release_combo.set(releases[0])

        self.branch_combo['values'] = branches
        if branches:
            self.branch_combo.set(branches[0])

        self.status_var.set("Ready")
        self._hide_progress_bar()

    def _on_release_selected(self, event=None):
        """
        Event handler for when a release is selected from the dropdown.
        Automatically selects the 'Specific Release' radio button.
        """
        self.latest_release_var.set("specific")

    def _on_branch_selected(self, event=None):
        """
        Event handler for when a branch is selected from the dropdown.
        Automatically selects the 'Development Branch' radio button.
        """
        self.latest_release_var.set("branch")

    def _clear_releases_cache(self):
        """
        Clear the cached releases and branches data.
        This can be called if we need to force a fresh fetch.
        """
        self._releases_cache = None
        self._branches_cache = None

    def _show_extras_selection_screen(self):
        """
        Show the extras selection screen.
        """
        self._clear_content()

        # Clear any cached extras data to ensure fresh loading
        if hasattr(self, 'available_extras'):
            delattr(self, 'available_extras')
        if hasattr(self, 'recommended_extras'):
            delattr(self, 'recommended_extras')
        if hasattr(self, 'extra_vars'):
            delattr(self, 'extra_vars')

        # Title
        title_label = ttk.Label(self.content_frame, text="Select Installation Extras",
                                style="Title.TLabel")
        title_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        # Description
        desc_label = ttk.Label(self.content_frame,
                               text="Select the additional components you want to install. Recommended extras are pre-selected based on your system.",
                               wraplength=500, font=("Arial", 9))
        desc_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        # Extras frame
        extras_frame = ttk.LabelFrame(self.content_frame, text="Available Extras", padding="10")
        extras_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        extras_frame.columnconfigure(0, weight=1)

        # Create scrollable frame for extras
        canvas = tk.Canvas(extras_frame)
        scrollbar = ttk.Scrollbar(extras_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Load extras (always fresh based on current source directory)
        self._load_extras(scrollable_frame)

        # Pack canvas and scrollbar
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        extras_frame.rowconfigure(0, weight=1)

        # Don't reset the screens array here - keep the current flow
        self.current_screen = 2
        self._update_navigation()

    def _load_extras(self, parent_frame):
        """Load and display available extras."""
        try:
            # Check global state for source directory
            if not self._has_valid_source_directory():
                # Show message when source directory is not available
                no_extras_label = ttk.Label(parent_frame,
                                            text="No source directory available. Please go back and download the source first.",
                                            font=("Arial", 10), foreground="red")
                no_extras_label.grid(row=0, column=0, columnspan=2, pady=20)
                return

            # Load setup.py
            setup_path = os.path.join(self.source_dir, 'setup.py')

            # Load setup.py and capture detailed error information
            error_details = []

            def capture_error(msg):
                error_details.append(msg)
                print(f"SETUP_DEBUG: {msg}")  # Also print to console

            capture_error(f"Attempting to load setup.py from: {setup_path}")
            capture_error(f"setup.py exists: {os.path.exists(setup_path)}")

            if os.path.exists(setup_path):
                try:
                    # Test reading the file
                    with open(setup_path, 'r', encoding='utf-8') as f:
                        content_preview = f.read(500)  # First 500 chars
                    capture_error(f"setup.py readable, size: {len(content_preview)} chars")
                except Exception as e:
                    capture_error(f"Cannot read setup.py file: {e}")

            # Create a custom log callback that captures errors
            def debug_log_callback(msg):
                capture_error(f"ANALYZER: {msg}")

            setup_analyzer = SetupAnalyzer(setup_path, log_callback=debug_log_callback)
            capture_error("SetupAnalyzer initialized")

            # Capture any exception during setup.py loading for better error reporting
            try:
                capture_error("Calling setup_analyzer.load_setup_as_library()...")
                success = setup_analyzer.load_setup_as_library()
                capture_error(f"load_setup_as_library() returned: {success}")
            except Exception as e:
                # Log the exception that SetupAnalyzer might not have caught
                capture_error(f"EXCEPTION during setup.py loading: {e}")
                capture_error(f"Full traceback:\n{traceback.format_exc()}")
                success = False

            if not success:
                # Create a detailed error display right on this page
                error_label = ttk.Label(parent_frame,
                                        text="Failed to load setup.py - see details below:",
                                        font=("Arial", 10, "bold"), foreground="red")
                error_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)

                # Create scrollable text widget to show error details
                error_frame = ttk.Frame(parent_frame)
                error_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

                error_text = tk.Text(error_frame, height=15, wrap=tk.WORD, font=("Consolas", 9))
                error_scrollbar = ttk.Scrollbar(error_frame, orient="vertical", command=error_text.yview)
                error_text.configure(yscrollcommand=error_scrollbar.set)

                # Add all captured error details
                error_content = "\n".join(error_details)
                error_text.insert(tk.END, error_content)
                error_text.config(state=tk.DISABLED)

                error_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
                error_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

                error_frame.columnconfigure(0, weight=1)
                error_frame.rowconfigure(0, weight=1)
                parent_frame.rowconfigure(1, weight=1)

                # Add error explanation
                error_explanation = ttk.Label(parent_frame,
                                              text="⚠️ Installation cannot proceed without loading setup.py. Please report this issue.",
                                              font=("Arial", 9, "bold"), foreground="red")
                error_explanation.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)

                return

            # Detect GPU information for intelligent extra recommendations
            gpu_info = detect_gpu()

            self.available_extras = setup_analyzer.get_available_extras()
            self.recommended_extras = setup_analyzer.get_recommended_extras(gpu_info)
            extra_descriptions = setup_analyzer.get_extra_descriptions()

            # Create checkboxes for each extra
            self.extra_vars = {}
            row = 0

            if not self.available_extras:
                # Show message when no extras are available
                no_extras_label = ttk.Label(parent_frame,
                                            text="No extras available for this platform/version.",
                                            font=("Arial", 10))
                no_extras_label.grid(row=0, column=0, columnspan=2, pady=20)
                return

            for extra_name, extra_deps in self.available_extras.items():
                # Checkbox
                var = tk.BooleanVar(value=extra_name in self.recommended_extras)
                self.extra_vars[extra_name] = var

                checkbox = ttk.Checkbutton(parent_frame, text=extra_name, variable=var)
                checkbox.grid(row=row, column=0, sticky=tk.W, pady=2)

                # Description
                desc = extra_descriptions.get(extra_name, "No description available")
                desc_label = ttk.Label(parent_frame, text=desc, wraplength=300,
                                       font=("Arial", 9))
                desc_label.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))

                row += 1

            parent_frame.columnconfigure(1, weight=1)

        except Exception as e:
            # If any error occurs during extras loading, show a fallback interface
            error_msg = f"Error loading extras: {e}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"EXTRAS_LOADING_ERROR: {error_msg}")  # Print to console

            # Create a simple error display
            error_label = ttk.Label(parent_frame,
                                    text="Error loading extras. Installation will continue with default settings.",
                                    font=("Arial", 10), foreground="red")
            error_label.grid(row=0, column=0, columnspan=2, pady=20)

            # Create a text widget to show the error details
            error_frame = ttk.Frame(parent_frame)
            error_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

            error_text = tk.Text(error_frame, height=10, wrap=tk.WORD)
            error_scrollbar = ttk.Scrollbar(error_frame, orient="vertical", command=error_text.yview)
            error_text.configure(yscrollcommand=error_scrollbar.set)

            error_text.insert(tk.END, error_msg)
            error_text.config(state=tk.DISABLED)

            error_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            error_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

            error_frame.columnconfigure(0, weight=1)
            error_frame.rowconfigure(0, weight=1)
            parent_frame.rowconfigure(1, weight=1)

    def _show_installation_screen(self):
        """
        Show the installation screen.
        """
        self._clear_content()

        # Title
        title_label = ttk.Label(self.content_frame, text="Installation Progress",
                                style="Title.TLabel")
        title_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        # Status - use global status_var so it appears above back/close buttons
        self.status_var.set("Preparing for installation...")
        # Optional: Keep local status display within content for visual consistency
        local_status_var = tk.StringVar(value="Preparing for installation...")
        status_label = ttk.Label(self.content_frame, textvariable=local_status_var,
                                 font=("Arial", 9))
        status_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        # Store reference to update both global and local status
        self.local_status_var = local_status_var

        # Installation info removed to prevent UI overflow issues

        # Log text area with better layout
        log_frame = ttk.LabelFrame(self.content_frame, text="Installation Log", padding="10")
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 20))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        # Configure the main content frame to expand properly
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.rowconfigure(2, weight=1)

        self.log_text = tk.Text(log_frame, height=15, width=80)
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Save log button
        save_log_button = ttk.Button(log_frame, text="Save Log to File", command=self._save_log_manually)
        save_log_button.grid(row=1, column=0, pady=(10, 0))

        # Don't reset the screens array here - keep the current flow
        self.current_screen = 4
        self._update_navigation()

        # Disable Next button during installation and show "Please wait..."
        self.next_button.config(state="disabled", text="Please wait...")

        # Start installation
        self._start_installation()

    def _update_install_status(self, message):
        """Update both global and local status variables during installation."""
        self.status_var.set(message)
        if hasattr(self, 'local_status_var'):
            self.local_status_var.set(message)

    def _update_uninstall_status(self, message):
        """Update both global and local status variables during uninstallation."""
        self.status_var.set(message)
        if hasattr(self, 'local_uninstall_status_var'):
            self.local_uninstall_status_var.set(message)

    def _start_installation(self):
        """Start the installation process."""

        def install():
            try:
                self._log("Starting dgenerate installation...")
                self._update_install_status("Installing...")

                # Create installer with logging callback
                installer = UvInstaller(log_callback=self._log, source_dir=self.source_dir)

                # Load setup.py early to get torch version and python requirements
                self._log("Analyzing dgenerate setup...")
                if not installer.setup_analyzer.load_setup_as_library():
                    self._log("Failed to load setup.py")
                    self._update_install_status("Setup analysis failed")
                    return
                installer.mark_setup_analyzed()

                # Skip Python compatibility check since uv will handle Python installation
                self._log("Skipping Python version compatibility check (uv will install required Python version)...")

                # Get PyTorch index URL based on torch version (now available after setup analysis)
                torch_version = installer.setup_analyzer.get_torch_version()
                self._log(f"DEBUG: Extracted torch version from setup.py: {torch_version}")
                torch_index_url = get_torch_index_url(torch_version)
                if torch_index_url:
                    self._log(f"Using PyTorch index: {torch_index_url}")
                    if torch_version:
                        self._log(f"Based on torch version: {torch_version}")
                else:
                    self._log("No PyTorch index URL determined")

                # Handle existing installation choice if we have one
                if hasattr(self, 'existing_install_choice') and hasattr(self, 'existing_install_data'):
                    choice = self.existing_install_choice.get()
                    if choice == 'cancel':
                        self._log("Installation cancelled by user")
                        self._update_install_status("Installation cancelled")
                        return
                    elif choice == 'uninstall':
                        self._log("Uninstalling existing installation...")
                        if not installer.uninstall_completely():
                            self._log("Failed to uninstall existing installation")
                            self._update_install_status("Uninstall failed")
                            return
                        self._log("Existing installation uninstalled successfully")
                    elif choice == 'overwrite':
                        self._log("Proceeding with overwrite of existing installation")
                        installer.cleanup_existing_installations()

                # Install (skip existing check since we already handled it)
                result = installer.install(
                    self.selected_extras,
                    torch_index_url,
                    commit_hash=self.commit_hash,
                    branch=self.branch_name,
                    is_pre_release=self.is_pre_release,
                    skip_existing_check=True
                )

                if result.success:
                    self._log("Installation completed successfully!")
                    self._update_install_status("Install Complete")

                    # Add completion info to log instead of showing separate screen
                    self._log("\n" + "=" * 60)
                    self._log("INSTALLATION COMPLETE!")
                    self._log("=" * 60)
                    self._log("\nWhat's next:")

                    # Dynamic completion message based on what was actually created
                    if result.desktop_shortcut_created:
                        self._log("• A desktop shortcut has been created to launch dgenerate Console")
                        self._log("• dgenerate has been added to your system PATH")
                        self._log("• You can now run 'dgenerate' from any terminal")
                        self._log("\nTo get started:")
                        self._log("1. Double-click the desktop shortcut (launches Console), or")
                        self._log("2. Open a terminal and run: 'dgenerate --console' or 'dgenerate --help'")
                    else:
                        self._log("• dgenerate has been added to your system PATH")
                        self._log("• You can now run 'dgenerate' from any terminal")
                        self._log("\nTo get started:")
                        self._log("1. Open a terminal and run: dgenerate --help")

                    self._log("\nFor more information, visit: https://github.com/Teriks/dgenerate")
                    self._log("=" * 60)

                    # Remove cancel button and change next to finish
                    self.root.after(0, self._on_installation_complete)
                else:
                    self._log("Installation failed!")
                    if result.error:
                        self._log(f"Error: {result.error}")
                    self._update_install_status("Installation failed")
                    # Update button for failed installation
                    self.root.after(0, self._on_installation_failed)


            except Exception as e:
                error_msg = f"Installation error: {e}\n\nFull traceback:\n{traceback.format_exc()}"
                self._log(error_msg)
                self._update_install_status("Installation failed")

                # Save log to file
                log_path = self._save_log_to_file()
                if log_path:
                    error_dialog_msg = f"Installation failed: {e}\n\nLog saved to: {log_path}\n\nCheck the log for details."
                else:
                    error_dialog_msg = f"Installation failed: {e}\n\nCheck the log for details."

                self._show_centered_messagebox('error', "Installation Error", error_dialog_msg)
                # Update button for failed installation
                self.root.after(0, self._on_installation_failed)

        threading.Thread(target=install, daemon=True).start()

    def _on_installation_complete(self):
        """
        Handle UI updates when installation is complete.
        """
        # Change next button to close and disable back button to prevent state issues
        self.back_button.config(state="disabled")  # Prevent going back after completion
        self.next_button.config(state="normal", text="Close")

        # Update navigation state but stay on install screen with log
        # Don't reset the screens array here - keep the current flow
        self.current_screen = 4  # Set to complete state (index 4 = 'complete')

    def _on_installation_failed(self):
        """
        Handle UI updates when installation fails.
        """
        # Enable back button to allow going back
        self.back_button.config(state="normal")
        # Change next button to close
        self.next_button.config(state="normal", text="Close")

    def _on_uninstallation_failed(self):
        """
        Handle UI updates when uninstallation fails.
        """
        # Enable back button to allow going back or retrying
        self.back_button.config(state="normal")
        # Change next button to close
        self.next_button.config(state="normal", text="Close")

    def _log(self, message: str):
        """
        Add a message to the log.

        :param message: Message to log
        """

        def update_log():
            try:
                # Check if log_text widget still exists and is valid
                if hasattr(self, 'log_text') and self.log_text.winfo_exists():
                    self.log_text.insert(tk.END, f"{message}\n")
                    self.log_text.see(tk.END)
                    self.root.update_idletasks()
            except tk.TclError:
                # Widget was destroyed or is invalid, just ignore the update
                pass
            except Exception as e:
                # Other errors, print to console for debugging
                print(f"Error updating log: {e}")

        # Also print to console for debugging
        print(message)

        try:
            self.root.after(0, update_log)
        except tk.TclError:
            # Root window might be destroyed, just print to console
            pass

    def _save_log_to_file(self):
        """
        Save the current log to a file.

        :return: Path to the saved log file or None if failed
        """
        try:
            # Create log filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"dgenerate_installer_log_{timestamp}.txt"

            # Save to temp directory
            log_path = os.path.join(tempfile.gettempdir(), log_filename)

            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(self.log_text.get(1.0, tk.END))

            return log_path
        except Exception as e:
            print(f"Failed to save log to file: {e}")
            return None

    def _save_log_manually(self):
        """
        Manually save the log to a file chosen by the user.
        """
        try:
            # Create default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"dgenerate_installer_log_{timestamp}.txt"

            # Ask user where to save
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=default_filename,
                title="Save Installation Log"
            )

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get(1.0, tk.END))

                self._show_centered_messagebox('info', "Log Saved", f"Installation log saved to:\n{file_path}")
        except Exception as e:
            self._show_centered_messagebox('error', "Error", f"Failed to save log: {e}")

    def _clear_content(self):
        """
        Clear the content frame.
        """
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def _update_navigation(self):
        """
        Update navigation button states and text based on current screen and flow.
        """
        # Determine the current flow and screen
        current_flow = self._get_current_flow()
        current_screen_name = self._get_current_screen_name()

        # Update button states and text based on flow and screen
        self._update_buttons_for_flow(current_flow, current_screen_name)

    def _get_current_flow(self):
        """Determine the current flow based on screens array."""
        if len(self.screens) == 3 and self.screens[0] == 'existing_install' and self.screens[1] == 'uninstall' and \
                self.screens[2] == 'uninstall_complete':
            return 'uninstall_flow'
        elif len(self.screens) == 2 and self.screens[0] == 'existing_install' and self.screens[
            1] == 'uninstall_complete':
            return 'uninstall_flow'
        elif 'welcome' in self.screens and self.screens[0] == 'welcome':
            return 'normal_install_flow'
        else:
            return 'unknown_flow'

    def _get_current_screen_name(self):
        """Get the name of the current screen."""
        if self.current_screen < len(self.screens):
            return self.screens[self.current_screen]
        return 'unknown'

    def _update_buttons_for_flow(self, flow, screen_name):
        """Update button states and text based on flow and screen."""
        if flow == 'uninstall_flow':
            if screen_name == 'existing_install':
                # Existing installation choice screen
                self.back_button.pack(side=tk.LEFT, padx=(0, 10))  # Make sure back button is visible
                self.back_button.config(state="disabled")
                self.next_button.config(state="normal", text="Continue")
            elif screen_name == 'uninstall':
                # Uninstall in progress screen
                self.back_button.pack_forget()  # Hide back button during uninstall
                self.next_button.config(state="disabled", text="Please wait...")
            elif screen_name == 'uninstall_complete':
                # Uninstall completion screen
                self.back_button.pack_forget()  # Hide back button - no going back after completion
                self.next_button.config(state="normal", text="Close")
        elif flow == 'normal_install_flow':
            if screen_name == 'welcome':
                # Welcome screen
                self.back_button.pack(side=tk.LEFT, padx=(0, 10))  # Make sure back button is visible
                self.back_button.config(state="disabled")
                self.next_button.config(state="normal", text="Next")
            elif screen_name == 'install':
                # Installation in progress screen
                self.back_button.pack_forget()  # Hide back button completely
                self.next_button.config(state="disabled", text="Installing...")
            elif screen_name == 'complete':
                # Installation completion screen
                self.back_button.pack_forget()  # Hide back button - no going back after completion
                self.next_button.config(state="normal", text="Finish")
            else:
                # All other screens in normal flow
                self.back_button.pack(side=tk.LEFT, padx=(0, 10))  # Make sure back button is visible
                self.back_button.config(state="normal")
                self.next_button.config(state="normal", text="Next")
        else:
            # Fallback for unknown flows
            self.back_button.pack(side=tk.LEFT, padx=(0, 10))  # Make sure back button is visible
            self.back_button.config(state="normal")
            self.next_button.config(state="normal", text="Next")

    def _go_back(self):
        """
        Go to the previous screen.
        """
        if self.current_screen > 0:
            self.current_screen -= 1
            self._show_current_screen()
        elif (self.current_screen == 0 and 'existing_install' in self.screens and self.screens[
            0] == 'existing_install'):
            # If we're on the existing installation screen (first screen), disable back
            pass  # Back button should be disabled

    def _go_next(self):
        """
        Handle next button actions based on current flow and screen.
        """
        current_flow = self._get_current_flow()
        current_screen_name = self._get_current_screen_name()

        # Handle actions based on flow and screen
        self._handle_next_action(current_flow, current_screen_name)

    def _handle_next_action(self, flow, screen_name):
        """Handle next button actions based on flow and screen."""
        # Check if button text is "Close" - if so, close the app regardless of screen
        if self.next_button.cget("text") == "Close":
            self.root.quit()
            return

        if flow == 'uninstall_flow':
            if screen_name == 'existing_install':
                # Handle existing installation choice
                self._handle_existing_installation_choice()
            elif screen_name == 'uninstall':
                # Uninstall in progress - button should be disabled
                pass
            elif screen_name == 'uninstall_complete':
                # Close installer
                self.root.quit()
        elif flow == 'normal_install_flow':
            if screen_name == 'welcome':
                # Welcome -> Source selection
                self.current_screen = 1
                self._show_source_selection_screen()
            elif screen_name == 'source':
                # Source selection -> Download and then Extras
                if self._validate_source_selection():
                    # Download the source first
                    self._download_source()
                # Don't navigate yet - download will navigate when complete
            elif screen_name == 'extras':
                # Extras -> Installation
                self.current_screen = 3
                self._prepare_installation()
            elif screen_name == 'install':
                # Install -> Complete
                self.current_screen = 4
                self._show_installation_screen()
            elif screen_name == 'complete':
                # Complete -> Finish
                self._finish()
        else:
            # Fallback for unknown flows
            print(f"Unknown flow: {flow}, screen: {screen_name}")

    def _navigate_to_screen(self, screen_name):
        """Navigate to a specific screen by name."""
        if screen_name in self.screens:
            self.current_screen = self.screens.index(screen_name)
            self._show_current_screen()

    def _handle_existing_installation_choice(self):
        """Handle the user's choice on the existing installation screen."""
        if hasattr(self, 'existing_install_choice'):
            choice = self.existing_install_choice.get()
            if choice == 'cancel':
                self.root.quit()
            elif choice == 'uninstall':
                # Go directly to uninstallation
                self._start_uninstallation()
            elif choice == 'overwrite':
                # Switch to normal installation flow and show welcome screen
                # Source directory is global state, so it's automatically preserved
                self.screens = ['welcome', 'source', 'extras', 'install', 'complete']
                self.current_screen = 0
                self._show_welcome_screen()
            else:
                # Invalid choice, stay on current screen
                pass
        else:
            # No choice made, stay on current screen
            pass

    def _validate_source_selection(self) -> bool:
        """Validate the source selection."""
        source_type = self.latest_release_var.get()

        if source_type == "specific" and not self.release_var.get():
            self._show_centered_messagebox('error', "Error", "Please select a specific release.")
            return False
        elif source_type == "branch" and not self.branch_var.get():
            self._show_centered_messagebox('error', "Error", "Please select a branch.")
            return False

        return True

    def _download_source(self):
        """Download the selected source."""

        # Disable back button during download
        self.back_button.config(state="disabled")
        self.next_button.config(state="disabled", text="Downloading...")

        def download():
            try:
                self.status_var.set("Downloading source code...")
                self._show_progress_bar()
                self.progress_var.set(0)  # Start at 0%

                source_type = self.latest_release_var.get()

                if source_type == "latest":
                    ref = "master"
                    # For latest, we'll get the commit from the master branch
                    self.commit_hash = self._get_latest_commit("master")
                    self.branch_name = "master"
                    self.is_pre_release = False
                elif source_type == "specific":
                    release_name = self.release_var.get().split(" - ")[0]
                    ref = release_name
                    # For releases, get the commit from the release
                    self.commit_hash = self._get_release_commit(release_name)
                    self.branch_name = "master"  # Releases are typically from master
                    self.is_pre_release = self._is_pre_release(release_name)
                else:  # branch
                    ref = self.branch_var.get()
                    # For branches, get the latest commit from that branch
                    self.commit_hash = self._get_latest_commit(ref)
                    self.branch_name = ref
                    self.is_pre_release = True  # Branches are typically pre-release

                # Create temporary directory
                temp_dir = tempfile.mkdtemp(prefix="dgenerate_install_")

                # Progress callback for real download progress
                def progress_callback(downloaded, total):
                    if total > 0:
                        percentage = (downloaded / total) * 90  # Use 90% for download, 10% for extraction
                        self.root.after(0, lambda: self.progress_var.set(percentage))

                        # Update status with download info
                        downloaded_mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        status_text = f"Downloading... {downloaded_mb:.1f}MB / {total_mb:.1f}MB"
                        self.root.after(0, lambda: self.status_var.set(status_text))

                # Download source with progress tracking
                downloaded_source_dir = self.github_client.download_source_archive(ref, temp_dir, progress_callback)

                if downloaded_source_dir:
                    # Set source directory in global state (main thread)
                    def complete_download():
                        # Set global state - persists across all screens
                        self.source_dir = downloaded_source_dir

                        # Show extraction progress
                        self.progress_var.set(95)
                        self.status_var.set("Extracting...")

                        # Complete
                        self.progress_var.set(100)
                        self.status_var.set("Source downloaded successfully")
                        # Hide progress bar after download completes
                        self.root.after(1000, self._hide_progress_bar)  # Hide after 1 second

                        # Navigate to extras selection
                        self._navigate_to_screen('extras')

                    # Execute in main thread
                    self.root.after(0, complete_download)
                else:
                    def handle_download_failed():
                        self.status_var.set("Download failed")
                        self._hide_progress_bar()
                        # Re-enable buttons on failure
                        self.back_button.config(state="normal")
                        self.next_button.config(state="normal", text="Next")
                        self._show_centered_messagebox('error', "Download Failed",
                                                       "Failed to download source code. Please try again.")

                    self.root.after(0, handle_download_failed)

            except Exception as e:
                def handle_download_error():
                    self.status_var.set("Download failed")
                    self._hide_progress_bar()
                    # Re-enable buttons on error
                    self.back_button.config(state="normal")
                    self.next_button.config(state="normal", text="Next")
                    self._show_centered_messagebox('error', "Download Error", str(e))

                self.root.after(0, handle_download_error)

        threading.Thread(target=download, daemon=True).start()

    def _get_latest_commit(self, branch: str) -> str:
        """Get the latest commit hash for a branch."""
        try:
            # Get the latest commit from the branch
            commits = self.github_client.get_commits(branch, per_page=1)
            if commits:
                return commits[0]['sha'][:7]  # Short commit hash
            return "unknown"
        except Exception as e:
            print(f"Error getting latest commit for {branch}: {e}")
            return "unknown"

    def _get_release_commit(self, release_name: str) -> str:
        """Get the commit hash for a specific release."""
        try:
            # Get the release details (including pre-releases for lookup)
            releases = self.github_client.get_releases(include_prereleases=True)
            for release in releases:
                if release['tag_name'] == release_name:
                    return release['target_commitish'][:7]  # Short commit hash
            return "unknown"
        except Exception as e:
            print(f"Error getting release commit for {release_name}: {e}")
            return "unknown"

    def _is_pre_release(self, release_name: str) -> bool:
        """Check if a release is a pre-release."""
        try:
            releases = self.github_client.get_releases()
            for release in releases:
                if release['tag_name'] == release_name:
                    return release.get('prerelease', False)
            return False
        except Exception as e:
            print(f"Error checking pre-release status for {release_name}: {e}")
            return False

    def _prepare_installation(self):
        """Prepare for installation."""
        # Get selected extras
        self.selected_extras = [extra for extra, var in self.extra_vars.items() if var.get()]

        # Show installation screen
        self.current_screen = 4
        self._show_installation_screen()

    def _check_existing_installation_first(self):
        """Check for existing installation first, before showing any screens."""
        try:
            # Create a temporary installer to check for existing installations
            # Use a dummy source dir since we're just checking for existing installations
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_installer = UvInstaller(log_callback=lambda msg: None, source_dir=temp_dir)
                existing_install = temp_installer.detect_existing_installation()

                if existing_install['exists']:
                    # Store existing installation data in global state
                    self.existing_install_data = existing_install

                    # Show existing installation screen immediately
                    self.screens = ['existing_install', 'uninstall_complete']
                    self.current_screen = 0
                    self._show_existing_installation_screen(existing_install)
                else:
                    # No existing installation, show welcome screen and normal flow
                    self.screens = ['welcome', 'source', 'extras', 'install', 'complete']
                    self.current_screen = 0
                    self._show_welcome_screen()

        except Exception as e:
            # If check fails, show welcome screen and normal flow
            self.screens = ['welcome', 'source', 'extras', 'install', 'complete']
            self.current_screen = 0
            self._show_welcome_screen()

    def _check_existing_installation_immediate(self):
        """Check for existing installation immediately and show appropriate screen."""
        try:
            # Create a temporary installer to check for existing installations
            # Use a dummy source dir since we're just checking for existing installations
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_installer = UvInstaller(log_callback=lambda msg: None, source_dir=temp_dir)
                existing_install = temp_installer.detect_existing_installation()

                if existing_install['exists']:
                    # Show existing installation screen
                    self._show_existing_installation_screen(existing_install)
                else:
                    # No existing installation, proceed directly to source selection
                    self._show_source_selection_screen()

        except Exception as e:
            # If check fails, proceed to source selection
            self._show_source_selection_screen()

    def _check_existing_installation(self):
        """Check for existing installation and show appropriate screen."""

        def check():
            try:
                # Create a temporary installer to check for existing installations
                # Use a dummy source dir since we're just checking for existing installations
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_installer = UvInstaller(log_callback=lambda msg: None, source_dir=temp_dir)
                    existing_install = temp_installer.detect_existing_installation()

                    if existing_install['exists']:
                        # Show existing installation screen
                        self.root.after(0, lambda: self._show_existing_installation_screen(existing_install))
                    else:
                        # No existing installation, proceed directly to source selection
                        self.root.after(0, self._show_source_selection_screen)

            except Exception as e:
                # If check fails, proceed to source selection
                self.root.after(0, self._show_source_selection_screen)

        threading.Thread(target=check, daemon=True).start()

    def _start_uninstallation(self):
        """Start the uninstallation process directly."""
        # Show uninstall screen with log
        self._show_uninstall_screen()

        def uninstall():
            try:
                self._log("Starting dgenerate uninstallation...")
                self._update_uninstall_status("Uninstalling existing installation...")
                self._show_progress_bar()
                self.progress_var.set(0)

                # Create a custom progress callback for uninstall
                def progress_callback(step, total_steps, message):
                    progress_percent = (step / total_steps) * 100
                    self.root.after(0, lambda: self.progress_var.set(progress_percent))
                    self.root.after(0, lambda: self.status_var.set(message))
                    self._log(message)

                # Create a temporary installer to handle uninstallation
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_installer = UvInstaller(log_callback=self._log, source_dir=temp_dir)

                    # Perform uninstallation with progress updates
                    self._log("Removing dgenerate components...")
                    if temp_installer.uninstall_completely():
                        self._log("Uninstallation completed successfully!")
                        self.progress_var.set(100)
                        self._update_uninstall_status("Uninstallation completed successfully!")

                        # Show completion screen after a delay
                        self.root.after(2000, self._show_uninstall_completion_screen)
                    else:
                        self._log("ERROR: Uninstallation failed!")
                        self._update_uninstall_status("Uninstallation failed!")
                        self._hide_progress_bar()
                        # Update button states for failed uninstallation
                        self.root.after(0, self._on_uninstallation_failed)

            except Exception as e:
                self._log(f"ERROR: Exception during uninstallation: {e}")
                self._log(f"Traceback: {traceback.format_exc()}")
                self._update_uninstall_status("Uninstallation failed!")
                self._hide_progress_bar()
                # Update button states for failed uninstallation
                self.root.after(0, self._on_uninstallation_failed)

        threading.Thread(target=uninstall, daemon=True).start()

    def _show_uninstall_screen(self):
        """Show the uninstallation screen with log."""
        self._clear_content()

        # Title
        title_label = ttk.Label(self.content_frame, text="Uninstalling dgenerate",
                                style="Title.TLabel")
        title_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        # Status - use global status_var so it appears above back/close buttons
        self.status_var.set("Preparing for uninstallation...")
        # Optional: Keep local status display within content for visual consistency
        local_status_var = tk.StringVar(value="Preparing for uninstallation...")
        status_label = ttk.Label(self.content_frame, textvariable=local_status_var,
                                 font=("Arial", 9))
        status_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        # Store reference to update both global and local status
        self.local_uninstall_status_var = local_status_var

        # Progress bar (initially hidden)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.content_frame, variable=self.progress_var,
                                            maximum=100)
        # Don't grid it yet - _show_progress_bar will do that

        # Log area
        log_frame = ttk.LabelFrame(self.content_frame, text="Uninstallation Log", padding="10")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        # Create text widget with scrollbar
        self.log_text = tk.Text(log_frame, height=15, width=60, wrap=tk.WORD,
                                font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Configure content frame to expand
        self.content_frame.rowconfigure(3, weight=1)

        # Update navigation
        self.screens = ['existing_install', 'uninstall', 'uninstall_complete']
        self.current_screen = 1
        self._update_navigation()

    def _show_uninstall_completion_screen(self):
        """Show the uninstallation completion screen."""
        self._clear_content()

        # Title
        title_label = ttk.Label(self.content_frame, text="Uninstallation Complete",
                                style="Title.TLabel")
        title_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        # Success message
        success_label = ttk.Label(self.content_frame,
                                  text="The existing dgenerate installation has been successfully removed from your system.",
                                  wraplength=500, font=("Arial", 9))
        success_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        # What was removed
        removed_frame = ttk.LabelFrame(self.content_frame, text="Components Removed", padding="10")
        removed_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        removed_frame.columnconfigure(0, weight=1)

        removed_text = \
            """• Virtual environment
               • UV package manager
               • dgenerate executable
               • PATH integration
               • Desktop shortcuts
               • File associations
               • Installation directory"""

        removed_label = ttk.Label(removed_frame,
                                  text=inspect.cleandoc(removed_text),
                                  justify=tk.LEFT,
                                  font=("Arial", 9))
        removed_label.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Update navigation
        self.screens = ['existing_install', 'uninstall', 'uninstall_complete']
        self.current_screen = 2
        self._update_navigation()

    def _show_existing_installation_screen(self, existing_install=None):
        """Show the existing installation detection screen."""
        self._clear_content()

        # Options frame
        options_frame = ttk.LabelFrame(self.content_frame, text="What would you like to do?", padding="10")
        options_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        options_frame.columnconfigure(0, weight=1)

        # Variable to store user choice
        self.existing_install_choice = tk.StringVar(value="overwrite")

        # Radio buttons
        uninstall_radio = ttk.Radiobutton(options_frame,
                                          text="Uninstall completely - Remove all traces of the existing installation",
                                          variable=self.existing_install_choice, value="uninstall")
        uninstall_radio.grid(row=0, column=0, sticky=tk.W, pady=2)

        overwrite_radio = ttk.Radiobutton(options_frame,
                                          text="Overwrite - Keep existing installation but replace with new version",
                                          variable=self.existing_install_choice, value="overwrite")
        overwrite_radio.grid(row=1, column=0, sticky=tk.W, pady=2)

        cancel_radio = ttk.Radiobutton(options_frame,
                                       text="Cancel - Abort installation",
                                       variable=self.existing_install_choice, value="cancel")
        cancel_radio.grid(row=2, column=0, sticky=tk.W, pady=2)

        # Store existing installation data in global state
        self.existing_install_data = existing_install

        # Update navigation
        self._update_navigation()

    def _finish(self):
        """
        Finish the installation.
        """
        self.root.quit()

    def _on_window_close(self):
        """
        Handle window close event (cancellation).
        """
        if self._show_centered_messagebox('yesno', "Cancel Installation",
                                          "Are you sure you want to cancel the installation?"):
            # Clean up global state
            self._cleanup_global_state()
            self.root.destroy()

    def _cleanup_global_state(self):
        """
        Clean up global state and temporary files.
        """
        # Clean up temporary source directory
        if self.source_dir and os.path.exists(self.source_dir):
            try:
                shutil.rmtree(self.source_dir)
            except:
                pass

        # Clear global state
        self.source_dir = None
        self.existing_install_data = None

        # Clear cache
        self._releases_cache = None
        self._branches_cache = None

    def _has_valid_source_directory(self) -> bool:
        """
        Check if we have a valid source directory in global state.
        
        :return: True if source directory exists and is valid
        """
        return (self.source_dir is not None and
                os.path.exists(self.source_dir) and
                os.path.exists(os.path.join(self.source_dir, 'setup.py')))

    def _show_current_screen(self):
        """
        Show the current screen.
        """
        screen_name = self.screens[self.current_screen]

        if screen_name == 'welcome':
            self._show_welcome_screen()
        elif screen_name == 'source':
            self._show_source_selection_screen()
        elif screen_name == 'extras':
            self._show_extras_selection_screen()
        elif screen_name == 'existing_install':
            # Use global state for existing installation data
            if self.existing_install_data:
                self._show_existing_installation_screen(self.existing_install_data)
            else:
                # If no data available, go back to welcome screen
                self.screens = ['welcome', 'source', 'extras', 'install', 'complete']
                self.current_screen = 0
                self._show_welcome_screen()
        elif screen_name == 'uninstall':
            self._show_uninstall_screen()
        elif screen_name == 'uninstall_complete':
            self._show_uninstall_completion_screen()
        elif screen_name == 'install':
            self._show_installation_screen()
        elif screen_name == 'complete':
            # Stay on installation screen with log - completion info is added to log
            self._show_installation_screen()

    def run(self):
        """
        Run the GUI application.
        """
        self.root.mainloop()


def main():
    """
    Main entry point for the GUI application.
    """
    app = DGenerateInstallerGUI()
    app.run()


if __name__ == "__main__":
    main()
