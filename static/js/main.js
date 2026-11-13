document.addEventListener("DOMContentLoaded", () => {
    // ---------- Element refs ----------
    const uploadForm       = document.getElementById("upload-form");
    const videoFile        = document.getElementById("video-file");
    const fileLabel        = document.getElementById("file-label");
    const fileDrop         = document.getElementById("file-drop");
    const uploadBtn        = document.getElementById("upload-btn");
    const videoFeed        = document.getElementById("video-feed");
    const videoPlaceholder = document.getElementById("video-placeholder");
    const statusDot        = document.getElementById("status-dot");
    const statusText       = document.getElementById("status-text");
    const modelTag         = document.getElementById("model-tag");
    const countCow         = document.getElementById("count-cow");
    const countHealthy     = document.getElementById("count-healthy");
    const countLumpy       = document.getElementById("count-lumpy");

    // Results panel (video + JSON + CSV only, no download)
    const resultsCard  = document.getElementById("results-card");
    const fileVideo    = document.getElementById("file-video");
    const fileJson     = document.getElementById("file-json");
    const fileCsv      = document.getElementById("file-csv");
    const framesCount  = document.getElementById("frames-count");
    const peakCounts   = document.getElementById("peak-counts");

    // ---------- State ----------
    let pollTimer = null;
    let streamActive = false;
    let lastOutputShown = null;

    // ---------- Helpers ----------
    function setStatus(text, state = "idle") {
        statusText.textContent = text;
        statusDot.className = "dot";
        if (state === "active") statusDot.classList.add("active");
        if (state === "error")  statusDot.classList.add("error");
    }

    function basename(p) {
        if (!p) return "—";
        return p.split(/[\\/]/).pop();
    }

    function renderResults(out) {
        if (!out) {
            resultsCard.hidden = true;
            return;
        }
        const sig = JSON.stringify(out);
        if (lastOutputShown === sig) return;
        lastOutputShown = sig;

        resultsCard.hidden = false;

        fileVideo.textContent = basename(out.annotated_video);
        fileJson.textContent  = basename(out.summary_json);
        fileCsv.textContent   = basename(out.frames_csv);

        framesCount.textContent = out.frames_processed ?? 0;

        if (out.peak_counts) {
            const p = out.peak_counts;
            peakCounts.textContent =
                `Cow ${p.Cow} · Healthy ${p.Healthy} · Lumpy ${p.Lumpy}`;
        } else {
            peakCounts.textContent = "—";
        }
    }

    // ---------- File selection ----------
    fileDrop.addEventListener("click", () => videoFile.click());

    videoFile.addEventListener("change", () => {
        if (videoFile.files.length > 0) {
            fileLabel.textContent = videoFile.files[0].name;
            uploadBtn.disabled = false;
        }
    });

    // ---------- Drag & drop ----------
    ["dragenter", "dragover"].forEach(ev => {
        fileDrop.addEventListener(ev, e => {
            e.preventDefault();
            fileDrop.classList.add("dragover");
        });
    });
    ["dragleave", "drop"].forEach(ev => {
        fileDrop.addEventListener(ev, e => {
            e.preventDefault();
            fileDrop.classList.remove("dragover");
        });
    });
    fileDrop.addEventListener("drop", e => {
        const files = e.dataTransfer.files;
        if (!files || files.length === 0) return;
        try {
            const dt = new DataTransfer();
            for (const f of files) dt.items.add(f);
            videoFile.files = dt.files;
        } catch (_) {
            try { videoFile.files = files; } catch (_) {}
        }
        fileLabel.textContent = files[0].name;
        uploadBtn.disabled = false;
    });

    // ---------- Upload ----------
    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (videoFile.files.length === 0) return;

        uploadBtn.disabled = true;
        uploadBtn.textContent = "Uploading...";
        setStatus("Uploading", "idle");

        const formData = new FormData();
        formData.append("video", videoFile.files[0]);

        try {
            const res  = await fetch("/upload", { method: "POST", body: formData });
            const data = await res.json();

            if (!res.ok) {
                setStatus(data.error || "Upload failed", "error");
                uploadBtn.textContent = "Start Detection";
                uploadBtn.disabled = false;
                return;
            }

            setStatus("Processing", "active");
            resultsCard.hidden = true;
            lastOutputShown = null;
            startStream();
            startPolling();
        } catch (err) {
            console.error(err);
            setStatus("Network error", "error");
            uploadBtn.textContent = "Start Detection";
            uploadBtn.disabled = false;
        }
    });

    // ---------- Live stream ----------
    function startStream() {
        if (streamActive) return;
        streamActive = true;
        videoPlaceholder.style.display = "none";
        videoFeed.style.display = "block";
        videoFeed.src = "/video_feed?t=" + Date.now();
    }

    videoFeed.onerror = () => {
        if (streamActive) setStatus("Stream interrupted", "error");
    };

    // ---------- Polling ----------
    function startPolling() {
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(fetchAll, 1000);
        fetchAll();
    }

    async function fetchAll() {
        try {
            const cRes  = await fetch("/detection_counts");
            const cData = await cRes.json();
            const c     = cData.counts || {};

            countCow.textContent     = c.Cow     || 0;
            countHealthy.textContent = c.Healthy || 0;
            countLumpy.textContent   = c.Lumpy   || 0;

            if (cData.processing) {
                setStatus("Detecting", "active");
            } else {
                setStatus("Saved", "idle");
                uploadBtn.textContent = "Start Detection";
                uploadBtn.disabled = videoFile.files.length === 0;
                if (pollTimer) {
                    clearInterval(pollTimer);
                    pollTimer = null;
                }
            }

            const sRes  = await fetch("/status");
            const sData = await sRes.json();

            if (!sData.processing && sData.last_output) {
                renderResults(sData.last_output);
            }
        } catch (err) {
            // transient — keep polling
        }
    }

    // ---------- Initial model info ----------
    fetch("/status")
        .then(r => r.json())
        .then(data => {
            const lesion = data.lesion_model || "unknown";
            const cow    = basename(data.cow_model);
            modelTag.textContent = `Lesion: ${lesion} · Cow: ${cow}`;
            if (String(lesion).toLowerCase().includes("missing")) {
                modelTag.style.color = "#ffb347";
            }
        })
        .catch(() => {
            modelTag.textContent = "status unavailable";
        });

    setStatus("Ready", "idle");
});