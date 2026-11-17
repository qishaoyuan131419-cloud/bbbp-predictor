"use strict";

const form = document.querySelector("#prediction-form");
const smilesInput = document.querySelector("#smiles");
const predictButton = document.querySelector("#predict-button");
const buttonLabel = predictButton.querySelector(".button-label");
const errorMessage = document.querySelector("#error-message");
const emptyState = document.querySelector("#empty-state");
const resultContent = document.querySelector("#result-content");
const resultSubtitle = document.querySelector("#result-subtitle");
const resultSummary = document.querySelector(".result-summary");

const descriptors = [
  ["molecular_weight", "Molecular weight", "g/mol", 2],
  ["logp", "LogP", "", 2],
  ["tpsa", "TPSA", "A²", 2],
  ["hydrogen_bond_donors", "H bond donors", "", 0],
  ["hydrogen_bond_acceptors", "H bond acceptors", "", 0],
  ["rotatable_bonds", "Rotatable bonds", "", 0],
  ["ring_count", "Ring count", "", 0],
  ["heavy_atom_count", "Heavy atoms", "", 0],
  ["fraction_csp3", "Fraction Csp3", "", 2],
];

function setLoading(isLoading) {
  predictButton.disabled = isLoading;
  predictButton.classList.toggle("loading", isLoading);
  buttonLabel.textContent = isLoading ? "Calculating..." : "Predict";
  form.setAttribute("aria-busy", String(isLoading));
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function renderDescriptors(values) {
  const grid = document.querySelector("#descriptor-grid");
  grid.replaceChildren();

  for (const [key, label, unit, precision] of descriptors) {
    const item = document.createElement("div");
    item.className = "descriptor-item";

    const name = document.createElement("span");
    name.textContent = label;

    const value = document.createElement("strong");
    const numericValue = Number(values[key]);
    value.textContent = `${numericValue.toFixed(precision)}${unit ? ` ${unit}` : ""}`;

    item.append(name, value);
    grid.append(item);
  }
}

function renderResult(data) {
  const isPermeable = data.prediction === 1;
  const probability = Number(data.probability);
  const confidence = Number(data.confidence);

  emptyState.hidden = true;
  resultContent.hidden = false;
  resultSubtitle.textContent = "Prediction completed for the submitted molecule.";

  resultSummary.classList.toggle("negative", !isPermeable);
  document.querySelector("#result-kicker").textContent = isPermeable ? "Likely BBB positive" : "Likely BBB negative";
  document.querySelector("#result-label").textContent = isPermeable ? "Predicted to cross the BBB" : "Predicted not to cross the BBB";
  document.querySelector("#result-icon").textContent = isPermeable ? "+" : "−";
  document.querySelector("#probability").textContent = `${(probability * 100).toFixed(1)}%`;
  document.querySelector("#confidence").textContent = `${(confidence * 100).toFixed(1)}%`;
  document.querySelector("#probability-bar").style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
  document.querySelector("#molecular-formula").textContent = data.molecular_formula;
  document.querySelector("#canonical-smiles").textContent = data.canonical_smiles;
  document.querySelector("#structure-image").src = data.structure_image;
  renderDescriptors(data.descriptors);
}

async function submitPrediction(event) {
  event.preventDefault();
  errorMessage.hidden = true;

  const smiles = smilesInput.value.trim();
  if (!smiles) {
    showError("Enter a SMILES string before running the prediction.");
    smilesInput.focus();
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "The prediction request could not be completed.");
    }
    renderResult(data);
  } catch (error) {
    showError(error.message || "The server could not be reached.");
  } finally {
    setLoading(false);
  }
}

form.addEventListener("submit", submitPrediction);

document.querySelectorAll(".example").forEach((button) => {
  button.addEventListener("click", () => {
    smilesInput.value = button.dataset.smiles;
    smilesInput.focus();
  });
});

smilesInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    form.requestSubmit();
  }
});
