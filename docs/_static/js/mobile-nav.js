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

    // Off-screen sections stay at the 32rem placeholder until they are
    // close to the viewport. Their real height then pushes the hash
    // target off screen. Remember the measured height before scrolling.
    var lockedForWidth = 0;

    function clearSectionHeights() {
        document.querySelectorAll('.rst-content section').forEach(function (section) {
            section.style.containIntrinsicBlockSize = '';
            section.style.containIntrinsicHeight = '';
        });
        lockedForWidth = 0;
    }

    function contentBoxHeight(section) {
        var style = window.getComputedStyle(section);
        var extra = (parseFloat(style.paddingTop) || 0)
            + (parseFloat(style.paddingBottom) || 0)
            + (parseFloat(style.borderTopWidth) || 0)
            + (parseFloat(style.borderBottomWidth) || 0);
        return Math.max(0, section.offsetHeight - extra);
    }

    function lockSectionHeights(force) {
        if (!force && lockedForWidth === window.innerWidth) {
            return;
        }
        var sections = document.querySelectorAll('.rst-content section');
        if (!sections.length) {
            return;
        }
        sections.forEach(function (section) {
            section.style.contentVisibility = 'visible';
        });
        var heights = [];
        sections.forEach(function (section) {
            heights.push(contentBoxHeight(section));
        });
        sections.forEach(function (section, index) {
            var px = heights[index] + 'px';
            section.style.containIntrinsicBlockSize = px;
            section.style.containIntrinsicHeight = px;
            section.style.contentVisibility = '';
        });
        lockedForWidth = window.innerWidth;
    }

    function hashTarget() {
        if (!location.hash || window.matchMedia('(min-width: 769px)').matches) {
            return null;
        }
        var id = location.hash.slice(1);
        try {
            id = decodeURIComponent(id);
        } catch (error) {
            return null;
        }
        return document.getElementById(id);
    }

    function revealHash(force) {
        var target = hashTarget();
        if (!target || !target.scrollIntoView) {
            return;
        }
        lockSectionHeights(force);
        // Lay the target out even when its section is still skipped.
        var forced = [];
        var node = target;
        while (node && node !== document.documentElement) {
            if (window.getComputedStyle(node).contentVisibility === 'auto') {
                node.style.contentVisibility = 'visible';
                forced.push(node);
            }
            node = node.parentNode;
        }
        target.scrollIntoView();
        forced.forEach(function (el) {
            el.style.contentVisibility = '';
        });
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
        window.addEventListener('hashchange', function () {
            revealHash(false);
        });
        window.addEventListener('load', function () {
            revealHash(true);
        });
        window.addEventListener('resize', function () {
            if (!lockedForWidth || window.innerWidth === lockedForWidth) {
                return;
            }
            clearSectionHeights();
            revealHash(true);
        });
        revealHash(false);
        requestAnimationFrame(function () {
            revealHash(false);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bind);
    } else {
        bind();
    }
}());
