Console UI
==========

.. image:: https://raw.githubusercontent.com/Teriks/dgenerate-readme-embeds/master/ui5.gif
   :alt: console ui

You can launch a cross platform Tkinter GUI for interacting with a
live dgenerate process using ``dgenerate --console`` or via the optionally
installed desktop shortcut on Windows.

This provides a basic REPL for the dgenerate config language utilizing
a ``dgenerate --shell`` subprocess to act as the live interpreter, it
also features full context aware syntax highlighting for the dgenerate
config language.

It can be used to work with dgenerate without encountering the startup
overhead of loading large python modules for every command line invocation.

The GUI console supports command history via the up and down arrow keys as a
normal terminal would, optional multiline input for sending multiline commands / configuration
to the shell. And various editing niceties such as GUI file / directory path insertion,
the ability to insert templated command recipes for quickly getting started and getting results,
and a selection menu for inserting karras schedulers by name.

Also supported is the ability to view the latest image as it is produced by ``dgenerate`` or
``\image_process`` via an image pane or standalone window.

The image viewer features bounding box and coordinate selection which can be helpful for
interactive use, as well as loading arbitrary images, and a few other helpful things such as the
ability to show the current image file in the systems file explorer,
all via the right click context menu.

When the package extra ``console_ui_opengl`` is installed, zoom and pan operations
will be hardware accelerated for smooth operation. (Mouse Wheel or Ctrl+/Ctrl-),
(Alt+LeftClick or Middle Click), respectively.

The console UI always starts in single line entry mode (terminal mode), multiline input mode
is activated via the insert key and indicated by the presence of line numbers, you must deactivate this mode
to submit commands via the enter key, however you can use the run button from the run menu (or ``Ctrl+Space``)
to run code in this mode. You cannot page through command history in this mode, and code will remain in the
console input pane upon running it making the UI function more like a code editor than a terminal.

The console can be opened with a file loaded in multiline input mode
by using the command: ``dgenerate --console filename.dgen``

``Ctrl+Q`` can be used in input pane for killing and then restarting the background interpreter process.

``Ctrl+F`` (find) and ``Ctrl+R`` (find/replace) is supported for both the input and output panes.

All common text editing features that you would expect to find in a basic text editor are present,
as well as python regex support for find / replace, with group substitution supporting the syntax
``\n`` or ``\{n}`` where ``n`` is the match group number.

Scroll back history in the output window is currently limited to 10000 lines however the console
app itself echos all ``stdout`` and ``stderr`` of the interpreter, so you can save all output to a log
file via file redirection if desired when launching the console from the terminal.

This can be configured by setting the environmental variable ``DGENERATE_CONSOLE_MAX_SCROLLBACK=10000``

Command history is currently limited to 500 commands, multiline commands are also
saved to command history.  The command history file is stored at ``-/.dgenerate_console_history``,
on Windows this equates to ``%USERPROFILE%\.dgenerate_console_history``

This can be configured by setting the environmental variable ``DGENERATE_CONSOLE_MAX_HISTORY=500``

Any UI settings that persist on startup are stored in ``-/.dgenerate_console_settings`` or
on Windows ``%USERPROFILE%\.dgenerate_console_settings``

