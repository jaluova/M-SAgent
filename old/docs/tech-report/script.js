document.addEventListener("DOMContentLoaded", () => {
  const chips = document.querySelectorAll("[data-filter]");
  const cards = document.querySelectorAll("[data-tags]");
  let activeFilter = "all";

  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      activeFilter = chip.dataset.filter || "all";
      chips.forEach((item) => {
        item.classList.toggle("active", item === chip);
      });

      cards.forEach((card) => {
        const tags = (card.dataset.tags || "").split(/\s+/).filter(Boolean);
        const visible = activeFilter === "all" || tags.includes(activeFilter);
        card.hidden = !visible;
      });
    });
  });

  const lightbox = document.querySelector("[data-lightbox]");
  const lightboxImage = document.querySelector("[data-lightbox-image]");
  const lightboxCaption = document.querySelector("[data-lightbox-caption]");
  const lightboxClose = document.querySelector("[data-lightbox-close]");
  const triggers = document.querySelectorAll("[data-zoom-src]");

  function closeLightbox() {
    if (!lightbox) return;
    lightbox.setAttribute("aria-hidden", "true");
    if (lightboxImage) {
      lightboxImage.removeAttribute("src");
      lightboxImage.removeAttribute("alt");
    }
    if (lightboxCaption) {
      lightboxCaption.textContent = "";
    }
  }

  triggers.forEach((trigger) => {
    trigger.addEventListener("click", () => {
      if (!lightbox || !lightboxImage) return;
      lightboxImage.src = trigger.dataset.zoomSrc || "";
      lightboxImage.alt = trigger.dataset.zoomAlt || "";
      if (lightboxCaption) {
        lightboxCaption.textContent = trigger.dataset.zoomCaption || "";
      }
      lightbox.setAttribute("aria-hidden", "false");
    });
  });

  if (lightbox) {
    lightbox.addEventListener("click", (event) => {
      if (event.target === lightbox) {
        closeLightbox();
      }
    });
  }

  if (lightboxClose) {
    lightboxClose.addEventListener("click", closeLightbox);
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeLightbox();
    }
  });
});
