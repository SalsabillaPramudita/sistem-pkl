document.getElementById('predictionForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const keterampilan_teknis = document.getElementById('keterampilan_teknis').value;
    const preferensi_lokasi = document.getElementById('preferensi_lokasi').value;
    const program_studi = document.getElementById('program_studi').value;
    const industri = document.getElementById('industri').value;

    const formData = {
        keterampilan_teknis: keterampilan_teknis,
        preferensi_lokasi: preferensi_lokasi,
        program_studi: program_studi,
        industri: industri
    };

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('predictionResult');
        resultDiv.classList.remove('d-none');
        if (data.prediction) {
            resultDiv.textContent = `Hasil Prediksi: ${data.prediction}`;
            resultDiv.classList.add('alert-success');
        } else {
            resultDiv.textContent = "Terjadi kesalahan, coba lagi.";
            resultDiv.classList.add('alert-danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        const resultDiv = document.getElementById('predictionResult');
        resultDiv.classList.remove('d-none');
        resultDiv.textContent = "Terjadi kesalahan, coba lagi.";
        resultDiv.classList.add('alert-danger');
    });
});
