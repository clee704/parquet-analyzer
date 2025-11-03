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

document.querySelectorAll('.toggle-header').forEach(header => {
	const toggle = header.querySelector('.toggle-indicator');
	const content = header.nextElementSibling;

	const syncState = () => {
		const isOpen = content.classList.contains('is-open');
		toggle.textContent = isOpen ? '−' : '+';
		toggle.classList.toggle('is-open', isOpen);
	};

	syncState();

	header.addEventListener('click', () => {
		content.classList.toggle('is-open');
		syncState();
	});
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

document.querySelectorAll('.segment-link').forEach(link => {
	link.addEventListener('click', event => {
		const offset = link.dataset.segmentOffset;
		const targets = document.querySelectorAll('.segment[data-segment-offset="' + offset + '"]');
		if (targets.length === 0) return;
		const target = targets[targets.length - 1];
		const segmentId = target.id;
		location.hash = '#' + segmentId;
		onHashChange();
		event.preventDefault();
	});
});

function onHashChange() {
	const hash = location.hash;
	console.log("Hash changed to:", hash);
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
onHashChange();