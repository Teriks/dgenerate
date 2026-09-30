// The Read the Docs theme closes the mobile menu only for
// ".wy-menu-vertical .current ul li a". A top-level link is outside that
// selector, so the drawer stays open and covers the page.
(function () {
    function closeDrawer() {
        if (window.matchMedia('(min-width: 769px)').matches) {
            return;
        }
        document.querySelectorAll('[data-toggle="wy-nav-shift"]').forEach(function (node) {
            node.classList.remove('shift');
        });
    }

    function bind() {
        var menu = document.querySelector('.wy-menu-vertical');
        if (!menu) {
            return;
        }
        menu.addEventListener('click', function (event) {
            if (event.target.closest('.toctree-expand')) {
                return;
            }
            if (!event.target.closest('a')) {
                return;
            }
            closeDrawer();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bind);
    } else {
        bind();
    }
}());
