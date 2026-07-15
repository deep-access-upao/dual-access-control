(() => {
  const startButton = document.querySelector("#start-camera");
  const takeButton = document.querySelector("#take-photo");
  const video = document.querySelector("#camera");
  const canvas = document.querySelector("#canvas");
  const fileInput = document.querySelector("#capture");
  const status = document.querySelector("#camera-status");
  if (!startButton) return;

  let stream;
  startButton.addEventListener("click", async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      video.srcObject = stream;
      takeButton.disabled = false;
      status.textContent = "Cámara activa. Alinea el rostro y toma la captura.";
    } catch (error) {
      status.textContent = "No se pudo acceder a la cámara. Usa la carga de archivo.";
    }
  });

  takeButton.addEventListener("click", () => {
    if (!stream || !video.videoWidth) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const transfer = new DataTransfer();
      transfer.items.add(new File([blob], "captura-camara.jpg", { type: "image/jpeg" }));
      fileInput.files = transfer.files;
      status.textContent = "Captura lista para verificar.";
    }, "image/jpeg", 0.92);
  });

  window.addEventListener("beforeunload", () => {
    if (stream) stream.getTracks().forEach((track) => track.stop());
  });
})();
