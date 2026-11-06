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

    function addBackdrop() {
        if (document.querySelector('.dgenerate-nav-backdrop')) {
            return;
        }
        var backdrop = document.createElement('button');
        backdrop.type = 'button';
        backdrop.className = 'dgenerate-nav-backdrop';
        backdrop.hidden = true;
        backdrop.setAttribute('aria-label', 'Close contents');
        backdrop.addEventListener('click', function () {
            closeDrawer();
            syncDrawer();
        });
        document.body.appendChild(backdrop);
    }

    function syncDrawer() {
        var open = drawerOpen();
        document.body.classList.toggle('dgenerate-drawer-open', open);
        var backdrop = document.querySelector('.dgenerate-nav-backdrop');
        if (backdrop) {
            backdrop.hidden = !open;
        }
        var button = document.querySelector('.dgenerate-nav-toggle');
        if (button) {
            button.setAttribute('aria-expanded', open ? 'true' : 'false');
            button.setAttribute('aria-label', open ? 'Close contents' : 'Open contents');
        }
        if (open) {
            var current = document.querySelector('.wy-menu-vertical a.current');
            var scroller = document.querySelector('.wy-side-scroll');
            if (current && scroller) {
                var delta = current.getBoundingClientRect().top - scroller.getBoundingClientRect().top;
                scroller.scrollTop += delta - (scroller.clientHeight / 2);
            }
        }
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

        button.addEventListener('click', function () {
            themeToggle.click();
            syncDrawer();
        });
        document.body.appendChild(button);
        watchBanner(button);
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

    function revealHash() {
        if (!location.hash || window.matchMedia('(min-width: 769px)').matches) {
            return;
        }
        var id = location.hash.slice(1);
        try {
            id = decodeURIComponent(id);
        } catch (error) {
            return;
        }
        var target = document.getElementById(id);
        if (target && target.scrollIntoView) {
            target.scrollIntoView();
        }
    }

    function bind() {
        addBackdrop();
        addToggle();
        syncDrawer();
        var menu = document.querySelector('.wy-menu-vertical');
        if (menu) {
            menu.addEventListener('click', function (event) {
                if (event.target.closest('.toctree-expand')) {
                    return;
                }
                if (!event.target.closest('a')) {
                    return;
                }
                closeDrawer();
                syncDrawer();
            });
        }
        document.addEventListener('click', function (event) {
            if (!event.target.closest("[data-toggle='wy-nav-top']")) {
                return;
            }
            syncDrawer();
        });
        window.addEventListener('hashchange', revealHash);
        revealHash();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bind);
    } else {
        bind();
    }
}());
