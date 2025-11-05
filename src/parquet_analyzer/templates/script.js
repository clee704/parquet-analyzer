(() => {

function on(root, event, selector, handler) {
  root.addEventListener(event, (e) => {
    const target = e.target.closest(selector);
    if (target && root.contains(target)) handler(e, target);
  });
}

document.addEventListener('htmx:afterSwap', (e) => {
	const toggle = e.detail.elt.querySelector(".toggle-header");
	if (toggle) {
		toggle.click();
	}
});

const app = document.getElementById('app');

const byteToggles = Array.from(document.querySelectorAll('.toggle-bytes'));

const swapByteDisplays = () => {
	byteToggles.forEach(el => {
		const current = el.textContent;
		const alt = el.getAttribute('title');
		if (alt !== null) {
			el.textContent = alt;
			el.setAttribute('title', current);
		}
	});
};

byteToggles.forEach(el => {
	el.addEventListener('click', swapByteDisplays);
});

on(app, 'click', '.toggle-header', (e, header) => {
	const toggle = header.querySelector('.toggle-indicator');
	const content = header.nextElementSibling;
	const syncState = () => {
		if (content) {
			const isOpen = content.classList.contains('is-open');
			toggle.textContent = isOpen ? '−' : '+';
			toggle.classList.toggle('is-open', isOpen);
		}
	};
	if (content) {
		content.classList.toggle('is-open');
	}
	syncState();
});

document.querySelectorAll('.toggle-all').forEach(toggleAll => {
	const toggleExpand = toggleAll.querySelector(':scope > .toggle-expand-all');
	const toggleCollapse = toggleAll.querySelector(':scope > .toggle-collapse-all');
	const indicators = toggleAll.parentElement.querySelectorAll(':scope > * > .toggle-header .toggle-indicator');
	const contents = toggleAll.parentElement.querySelectorAll(':scope > * > .toggle-content');

	toggleExpand.addEventListener('click', () => {
		indicators.forEach(indicator => {
			indicator.textContent = '−';
			indicator.classList.add('is-open');
		});
		contents.forEach(content => content.classList.add('is-open'));
	});

	toggleCollapse.addEventListener('click', () => {
		indicators.forEach(indicator => {
			indicator.textContent = '+';
			indicator.classList.remove('is-open');
		});
		contents.forEach(content => content.classList.remove('is-open'));
	});
});

on(app, 'click', '.segment-link', (e, link) => {
	const findAndSetHash = () => {
		const offset = link.dataset.segmentOffset;
		const targets = document.querySelectorAll('.segment[data-segment-offset="' + offset + '"]');
		if (targets.length === 0) return;
		const target = targets[targets.length - 1];
		const segmentId = target.id;
		location.hash = '#' + segmentId;
		onHashChange();
	};
	if (!document.querySelector('.section--segments .segments')) {
		const toggle = document.querySelector('.section--segments > .toggle-header');
		document.body.addEventListener('htmx:afterSwap', function handler(e) {
			if (e.target.classList.contains('section--segments')) {
				findAndSetHash();
				document.body.removeEventListener('htmx:afterSwap', handler);
			}
		});
		toggle.click();
	} else {
		findAndSetHash();
	}
	e.preventDefault();
});

function onHashChange() {
	const hash = location.hash;
	if (!hash.startsWith('#segment-')) return;
	const segmentId = hash.substring(1);
	const target = document.getElementById(segmentId);
	if (!target) return;
	let current = target.querySelector(':scope > .toggle-content');
	if (!current) return;
	while (current) {
		if (current.classList.contains('toggle-content') &&
				!current.classList.contains("is-open") &&
				current.previousElementSibling &&
				current.previousElementSibling.classList.contains('toggle-header')) {
			current.previousElementSibling.click();
		}
		current = current.parentElement;
	}

	// Delay scrolling until after panels have opened so the element settles into place.
	setTimeout(() => {
		target.scrollIntoView({ behavior: 'smooth', block: 'start' });
	}, 0);
}

window.addEventListener('hashchange', onHashChange);

if (document.querySelector(".file-droppable")) {
	let dragCounter = 0;
	document.addEventListener('dragover', e => e.preventDefault());
	document.addEventListener('dragenter', e => {
		e.preventDefault();
		dragCounter++;
		document.body.classList.add("dragover");
	});
	document.addEventListener('dragleave', e => {
		e.preventDefault();
		dragCounter--;
		if (dragCounter === 0) {
			document.body.classList.remove("dragover");
		}
	});
	document.addEventListener("drop", e => handleDrop(e));
	async function handleDrop(e) {
		e.preventDefault();
		const file = e.dataTransfer?.files?.[0];
		uploadFile(file);
	}
}

})();

async function uploadFile(file) {
	document.body.classList.add("loading");
	try {
		const res = await fetch(`/upload?name=${encodeURIComponent(file.name)}`, {
			method: 'PUT',
			headers: {'Content-Type': 'application/octet-stream'},
			body: file
		});
		const html = await res.text();
		document.open();
		document.write(html);
		document.close();
	} finally {
		document.body.classList.remove("loading");
	}
}