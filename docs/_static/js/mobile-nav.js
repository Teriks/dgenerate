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

    function drawerOpen() {
        var node = document.querySelector('[data-toggle="wy-nav-shift"]');
        return !!(node && node.classList.contains('shift'));
    }

    function addToggle() {
        if (document.querySelector('.dgenerate-nav-toggle')) {
            return;
        }
        var themeToggle = document.querySelector("[data-toggle='wy-nav-top']");
        if (!themeToggle) {
            return;
        }
        var button = document.createElement('button');
        button.type = 'button';
        button.className = 'dgenerate-nav-toggle';
        button.innerHTML = '<span></span><span></span><span></span>';

        function label() {
            var open = drawerOpen();
            button.setAttribute('aria-expanded', open ? 'true' : 'false');
            button.setAttribute('aria-label', open ? 'Close contents' : 'Open contents');
        }

        button.addEventListener('click', function () {
            themeToggle.click();
            label();
        });
        document.body.appendChild(button);
        label();
        watchBanner(button);
        return label;
    }

    function watchBanner(button) {
        var banner = document.querySelector('.wy-nav-top');
        if (!banner || typeof IntersectionObserver === 'undefined') {
            return;
        }
        function apply(visible) {
            button.classList.toggle('banner-visible', visible);
        }
        var observer = new IntersectionObserver(function (entries) {
            apply(entries[0].isIntersecting);
        });
        observer.observe(banner);
        var rect = banner.getBoundingClientRect();
        apply(rect.bottom > 0 && rect.top < window.innerHeight && rect.width > 0);
    }

    function bind() {
        var menu = document.querySelector('.wy-menu-vertical');
        var label = addToggle();
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
            if (label) {
                label();
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bind);
    } else {
        bind();
    }
}());
