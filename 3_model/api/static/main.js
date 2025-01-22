// Define functions in global scope
let showPage;
let checkSimilarity;
let fileUpload;

// Define base API URL
const API_BASE_URL = 'http://localhost:8000'; // Change this based on your environment

document.addEventListener('DOMContentLoaded', function() {
    showPage = function(pageId) {
        document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
        document.getElementById(pageId).classList.add('active');
    };

    function showResult(message, type) {
        const result = document.getElementById('result');
        result.textContent = message;
        result.className = type;
        result.style.display = 'block';
    }

    function getSimilarityClass(score) {
        if (score >= 90) return 'similarity-90-100';
        if (score >= 80) return 'similarity-80-90';
        if (score >= 70) return 'similarity-70-80';
        if (score >= 60) return 'similarity-60-70';
        if (score >= 50) return 'similarity-50-60';
        return 'similarity-0-50';
    }

    function createLegend() {
        return `
            <div class="similarity-legend">
                <h4>Similarity Score Legend:</h4>
                <div class="legend-item similarity-90-100">90-100% (Excellent)</div>
                <div class="legend-item similarity-80-90">80-89% (Very Good)</div>
                <div class="legend-item similarity-70-80">70-79% (Good)</div>
                <div class="legend-item similarity-60-70">60-69% (Fair)</div>
                <div class="legend-item similarity-50-60">50-59% (Poor)</div>
                <div class="legend-item similarity-0-50">&lt;50% (Very Poor)</div>
            </div>
        `;
    }

    function calculateSummaryStats(results) {
        const scores = results.map(r => r.similarity_score * 100);
        const average = scores.reduce((a, b) => a + b, 0) / scores.length;
        const belowThreshold = scores.filter(score => score < 70).length;
        
        return {
            average,
            belowThreshold
        };
    }

    // Define fileUpload in the proper scope
    fileUpload = async function() {
        const fileInput = document.getElementById('csvFile');
        const button = document.getElementById('uploadButton');
        const loading = document.getElementById('file-loading');
        const progressInfo = document.getElementById('progress-info');
        const resultsTable = document.getElementById('results-table');

        if (!fileInput.files.length) {
            alert('Please select a file first.');
            return;
        }

        const file = fileInput.files[0];
        const reader = new FileReader();

        reader.onload = async function(event) {
            const content = event.target.result;
            const encodedBlob = new Blob([content], { type: file.type || "text/csv" });

            const formData = new FormData();
            formData.append('file', encodedBlob, file.name);

            button.disabled = true;
            loading.style.display = 'inline-block';
            progressInfo.style.display = 'block';
            progressInfo.textContent = 'Uploading file...';
            resultsTable.innerHTML = '';

            try {
                const response = await fetch(`${API_BASE_URL}/process-file`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => null);
                    throw new Error(
                        errorData?.detail || 
                        `Upload failed with status: ${response.status} ${response.statusText}`
                    );
                }

                const data = await response.json();
                
                if (!data.results || !Array.isArray(data.results)) {
                    throw new Error('Invalid response format from server');
                }
                
                // Add legend first
                resultsTable.innerHTML = createLegend();
                
                // Create results table
                let tableHTML = `
                    <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
                        <thead>
                            <tr>
                                <th style="border: 1px solid #ddd; padding: 8px; background-color: #f8f9fa;">Source Text</th>
                                <th style="border: 1px solid #ddd; padding: 8px; background-color: #f8f9fa;">Translated Text</th>
                                <th style="border: 1px solid #ddd; padding: 8px; background-color: #f8f9fa;">Similarity Score</th>
                            </tr>
                        </thead>
                        <tbody>
                `;

                data.results.forEach(result => {
                    const sourceText = result.source_text || result.sentence1;
                    const translatedText = result.translated_text || result.sentence2;
                    const score = (result.similarity_score * 100).toFixed(2);
                    const similarityClass = getSimilarityClass(parseFloat(score));
                    
                    tableHTML += `
                        <tr class="results-row ${similarityClass}">
                            <td style="border: 1px solid #ddd; padding: 8px;">${sourceText}</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">${translatedText}</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">${score}%</td>
                        </tr>
                    `;
                });

                tableHTML += '</tbody></table>';
                resultsTable.innerHTML += tableHTML;

                // Add summary statistics
                const summaryStats = calculateSummaryStats(data.results);
                progressInfo.innerHTML = `
                    Processed ${data.results.length} pairs<br>
                    Average similarity: ${summaryStats.average.toFixed(2)}%<br>
                    Below 70%: ${summaryStats.belowThreshold} pairs
                `;
            } catch (error) {
                progressInfo.textContent = `Error: ${error.message}`;
                console.error('Error:', error);
            } finally {
                button.disabled = false;
                loading.style.display = 'none';
            }
        };
        
        reader.readAsText(file);
    };

    // Define checkSimilarity in the proper scope
    checkSimilarity = async function() {
        const sentence1 = document.getElementById('sentence1').value.trim();
        const sentence2 = document.getElementById('sentence2').value.trim();
        const button = document.getElementById('checkSimilarity');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');

        if (!sentence1 || !sentence2) {
            showResult('Please enter both sentences.', 'error');
            return;
        }

        button.disabled = true;
        loading.style.display = 'inline-block';
        result.style.display = 'none';

        try {
            const response = await fetch(`${API_BASE_URL}/similarity`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    sentence1: sentence1,
                    sentence2: sentence2
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                throw new Error(
                    errorData?.detail || 
                    `Request failed with status: ${response.status} ${response.statusText}`
                );
            }

            const data = await response.json();
            const score = (data.similarity_score * 100).toFixed(2);
            if (score > 75) {
                showResult(`Similarity Score: ${score}%`, 'success');
            }
            else {
                showResult(`Similarity Score: ${score}%`, 'error');
            }
        } catch (error) {
            showResult(`Error: ${error.message}`, 'error');
            console.error('Error:', error);
        } finally {
            button.disabled = false;
            loading.style.display = 'none';
        }
    };
});