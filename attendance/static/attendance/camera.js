/* Browser-owned camera; one bounded upload at a time, always released on stop. */
(() => {
    const root = document.getElementById('browser-camera');
    if (!root) return;
    const video = document.getElementById('camera-video');
    const overlay = document.getElementById('camera-overlay');
    const status = document.getElementById('camera-status');
    const start = document.getElementById('camera-start');
    const stop = document.getElementById('camera-stop');
    const placeholder = document.getElementById('camera-placeholder');
    const canvas = document.createElement('canvas');
    const csrf = root.querySelector('[name=csrfmiddlewaretoken]').value;
    let stream = null, active = false, saved = 0, generation = 0;
    let controller = null;
    const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
    function stopCamera(message) {
        active = false;
        generation += 1;
        controller?.abort();
        stream?.getTracks().forEach(track => track.stop());
        stream = null;
        video.srcObject = null;
        video.hidden = true;
        placeholder.hidden = false;
        overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
        start.disabled = false;
        stop.disabled = true;
        if (message) status.textContent = message;
    }
    function draw(results) {
        overlay.width = canvas.width;
        overlay.height = canvas.height;
        const ctx = overlay.getContext('2d');
        ctx.font = '16px sans-serif';
        ctx.lineWidth = 3;
        results.forEach(result => {
            const [x, y, w, h] = result.box;
            ctx.strokeStyle = result.recognized ? '#85e48b' : '#ff7777';
            ctx.fillStyle = ctx.strokeStyle;
            ctx.strokeRect(x, y, w, h);
            ctx.fillText(result.name + (result.recognized ? ' — Present' : ''), x, Math.max(20, y - 8));
        });
    }
    async function processFrames(run) {
        while (active && generation === run) {
            try {
                const scale = Math.min(1, 640 / video.videoWidth, 480 / video.videoHeight);
                canvas.width = Math.max(32, Math.round(video.videoWidth * scale));
                canvas.height = Math.max(32, Math.round(video.videoHeight * scale));
                canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
                const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
                if (!active || generation !== run) return;
                if (!blob) throw new Error('Could not read the camera frame.');
                const form = new FormData();
                form.append('frame', blob, 'frame.jpg');
                controller = new AbortController();
                const timeout = setTimeout(() => controller?.abort(), 120000);
                let response;
                try {
                    response = await fetch(root.dataset.endpoint, {
                        method: 'POST', body: form, credentials: 'same-origin',
                        headers: {'X-CSRFToken': csrf}, signal: controller.signal,
                    });
                } finally { clearTimeout(timeout); }
                if (!active || generation !== run) return;
                if (response.status === 429) {
                    status.textContent = 'Recognition engine is busy. Retrying…';
                    await sleep(2000);
                    continue;
                }
                if (response.redirected || response.status === 403) throw new Error('Session expired or permission denied. Refresh and sign in again.');
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Camera processing failed. Please retry.');
                if (root.dataset.mode === 'capture') {
                    if (data.saved) saved += 1;
                    status.textContent = `${saved} / ${root.dataset.target} samples saved. ${data.message}`;
                    if (saved >= Number(root.dataset.target)) {
                        stopCamera(`Capture complete: ${saved} samples saved. Return to enrollment and train the model.`);
                        return;
                    }
                } else {
                    draw(data.results);
                    const names = data.results.filter(result => result.recognized).map(result => result.name);
                    status.textContent = names.length ? `Present: ${names.join(', ')}` : 'No recognized student. Face the camera in good light.';
                }
            } catch (error) {
                if (generation !== run) return;
                stopCamera(error.name === 'AbortError' ? 'Processing timed out. Wait for the server, then try again.' : error.message);
                return;
            }
            await sleep(root.dataset.mode === 'capture' ? 450 : 1000);
        }
    }
    start.addEventListener('click', async () => {
        if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
            status.textContent = 'Camera access needs HTTPS or localhost and a supported browser.';
            return;
        }
        start.disabled = true;
        status.textContent = 'Allow camera access when your browser asks…';
        const run = ++generation;
        try {
            const camera = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640}, height: {ideal: 480}, facingMode: 'user'}, audio: false});
            if (generation !== run) { camera.getTracks().forEach(track => track.stop()); return; }
            stream = camera;
            video.srcObject = stream;
            video.hidden = false;
            placeholder.hidden = true;
            await video.play();
            if (generation !== run) return;
            active = true;
            stop.disabled = false;
            status.textContent = 'Camera started. First recognition may take a moment while the model loads…';
            processFrames(run);
        } catch (error) {
            if (generation !== run) return;
            stopCamera(error.name === 'NotAllowedError' ? 'Camera permission denied. Allow access in your browser, then retry.' : 'Cannot open the camera. Check that it is connected and not used by another app.');
        }
    });
    stop.addEventListener('click', () => stopCamera('Camera stopped.'));
    window.addEventListener('pagehide', () => stopCamera());
    document.addEventListener('visibilitychange', () => { if (document.hidden) stopCamera('Camera stopped because this tab is hidden.'); });
})();
