document.addEventListener("DOMContentLoaded", function() {
    const tableBody = document.querySelector("#txnTable tbody");
    const searchInput = document.getElementById("search");
    const filterSelect = document.getElementById("filter");
    const maxTransactions = 20;
    let liveTransactionsData = [];
    let transactionTypeChart = null; // Store chart instance
    let transactionPatternChart = null; // Store line chart instance
    let riskScorePieChart = null; // Store pie chart instance
    // Function to render the table
    function renderTable(dataToRender) {
        tableBody.innerHTML = "";
        dataToRender.forEach(txn => {
            const row = document.createElement("tr");
            if (txn.Status === "High Risk") row.classList.add("fraud");
            row.innerHTML = `
                <td>${txn.TransactionID}</td>
                <td>${txn.Date}</td>
                <td>${txn.Amount}</td>
                <td>${txn.Type}</td>
                <td>${txn.Parties}</td>
                <td>${txn.RiskScore}%</td>
                <td>${txn.Status}</td>
                <td>${txn.Anomalies}</td>
            `;
            tableBody.appendChild(row);
        });
    }

    // Function to update the bar chart
    function updateBarChart(dataToRender) {
        const typeCounts = {};
        dataToRender.forEach(txn => {
            typeCounts[txn.Type] = (typeCounts[txn.Type] || 0) + 1;
        });

        const labels = Object.keys(typeCounts);
        const counts = Object.values(typeCounts);

        if (transactionTypeChart) {
            transactionTypeChart.data.labels = labels;
            transactionTypeChart.data.datasets[0].data = counts;
            transactionTypeChart.update();
        } else {
            const ctx = document.getElementById('transactionTypeChart').getContext('2d');
            transactionTypeChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Number of Transactions by Type',
                        data: counts,
                        backgroundColor: 'rgba(54, 162, 235, 0.5)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: { display: true, text: 'Transactions' }
                        },
                        x: {
                            title: { display: true, text: 'Type' }
                        }
                    },
                }
            });
        }
    }

    // Function to update the line graph
    function updateLineChart(dataToRender) {
        const labels = dataToRender.map(txn => txn.TransactionID).reverse();
        const amounts = dataToRender.map(txn => parseFloat(txn.Amount.replace('₹', ''))).reverse();

        if (transactionPatternChart) {
            transactionPatternChart.data.labels = labels;
            transactionPatternChart.data.datasets[0].data = amounts;
            transactionPatternChart.update();
        } else {
            const ctx = document.getElementById('transactionPatternChart').getContext('2d');
            transactionPatternChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Transaction Amount (Most Recent)',
                        data: amounts,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: false,
                            title: { display: true, text: 'Amount (₹)' }
                        },
                        x: {
                            title: { display: true, text: 'Transaction ID' }
                        }
                    },
                }
            });
        }
    }
    function updateRiskScorePieChart(transaction) {
        const riskScore = transaction.RiskScore || 0;
        const transactionID = transaction.TransactionID;

        const ctx = document.getElementById('riskScorePieChart').getContext('2d');
        const pieData = [riskScore, 100 - riskScore];

        if (riskScorePieChart) {
            riskScorePieChart.data.datasets[0].data = pieData;
            riskScorePieChart.options.plugins.title.text = `Risk Score for Transaction ${transactionID}`;
            riskScorePieChart.update();
        } else {
            riskScorePieChart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: ['Risk Score', 'Remaining'],
                    datasets: [{
                        data: pieData,
                        backgroundColor: ['#dc3545', '#28a745'],
                        borderColor: '#fff',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: 'bottom' },
                        title: { display: true, text: `Risk Score for Transaction ${transactionID}` }
                    }
                }
            });
        }
    }

    // Function to fetch a single new transaction and add to the live feed
    async function fetchAndAddTransaction() {
        try {
            const res = await fetch("/transaction");
            const newTransaction = await res.json();
            
            liveTransactionsData.unshift(newTransaction);
            if (liveTransactionsData.length > maxTransactions) {
                liveTransactionsData.pop();
            }

            if (searchInput.value.length === 0) {
                renderTable(liveTransactionsData);
                updateBarChart(liveTransactionsData);
                updateLineChart(liveTransactionsData);
            }
        } catch (error) {
            console.error("Error fetching transaction:", error);
        }
    }

    // Event listener for the search input
    searchInput.addEventListener("input", async (e) => {
        const query = e.target.value.toLowerCase();
        if (query.length > 0) {
            try {
                const res = await fetch(`/search_data?q=${query}`);
                const searchResults = await res.json();
                renderTable(searchResults);
                updateBarChart(searchResults);
                updateLineChart(searchResults);
            } catch (error) {
                console.error("Error fetching search results:", error);
            }
        } else {
            renderTable(liveTransactionsData);
            updateBarChart(liveTransactionsData);
            updateLineChart(liveTransactionsData);
        }
    });

    // Event listener for the filter dropdown
    filterSelect.addEventListener("change", async (e) => {
        const filter = e.target.value;
        const query = searchInput.value.toLowerCase();
        
        let dataToFilter;
        if (query.length > 0) {
            const res = await fetch(`/search_data?q=${query}`);
            dataToFilter = await res.json();
        } else {
            dataToFilter = liveTransactionsData;
        }

        let filteredData = dataToFilter;
        if (filter !== "All") {
            filteredData = dataToFilter.filter(txn => txn.Status === filter);
        }
        renderTable(filteredData);
    });

    // Initial load and set interval for live updates
    fetchAndAddTransaction(); 
    setInterval(fetchAndAddTransaction, 5000); 
});