const videoElem = document.getElementById("qr-video");
const fileInput = document.getElementById("file-input");
const resultBox = document.getElementById("result");

const qrScanner = new QrScanner(videoElem, result => handleQRCode(result), {
  highlightScanRegion: true,
  highlightCodeOutline: true,
});

qrScanner.start();

// Handle File Upload
fileInput.addEventListener("change", e => {
  const file = e.target.files[0];
  if (!file) return;
  QrScanner.scanImage(file, { returnDetailedScanResult: true })
    .then(result => handleQRCode(result.data))
    .catch(err => showError("QR Code not detected in image"));
});

// Function to handle QR decode result
function handleQRCode(data) {
  console.log("Decoded QR:", data);

  try {
    // Example QR might contain: payee_vpa=abc@upi&merchant_code=1234
    const params = new URLSearchParams(data);
    const payee_vpa = params.get("payee_vpa");
    const merchant_code = params.get("merchant_code");

    if (!payee_vpa || !merchant_code) {
      showError("Invalid QR format");
      return;
    }

    fetch("/analyze_qr", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ payee_vpa, merchant_code })
    })
    .then(res => res.json())
    .then(res => {
      if (res.error) {
        showError(res.error);
      } else {
        showResult(res);
      }
    })
    .catch(() => showError("Error contacting server"));
  } catch (e) {
    showError("Invalid QR format");
  }
}

function showResult(data) {
  resultBox.style.display = "block";
  resultBox.classList.remove("error");
  resultBox.innerHTML = `
    <h3>QR Analysis Result</h3>
    <p><b>Payee VPA:</b> ${data.payee_vpa}</p>
    <p><b>Merchant Code:</b> ${data.merchant_code}</p>
    <p><b>Location:</b> ${data.location}</p>
    <p><b>Device:</b> ${data.device}</p>
    <p><b>Transaction Type:</b> ${data.transaction_type}</p>
  `;
}

function showError(msg) {
  resultBox.style.display = "block";
  resultBox.classList.add("error");
  resultBox.innerHTML = `<h3>Error</h3><p>${msg}</p>`;
}
