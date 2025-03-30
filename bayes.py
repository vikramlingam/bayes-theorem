import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter

# Set up the styling for better readability
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 20
})

def create_bayes_visualization(prior_prob, sensitivity, specificity, population=1000):
    """Create a clear, readable visualization of Bayes' theorem."""

    # Calculate values based on Bayes theorem
    true_positive = prior_prob * sensitivity * population
    false_positive = (1 - prior_prob) * (1 - specificity) * population
    true_negative = (1 - prior_prob) * specificity * population
    false_negative = prior_prob * (1 - sensitivity) * population

    # Calculate posterior probability (positive predictive value)
    posterior_prob = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

    # Create a large figure with a clean white background
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    gs = GridSpec(3, 2, height_ratios=[1, 2, 1], width_ratios=[1, 1])

    # Title for the entire visualization
    fig.suptitle("Bayes' Theorem: Medical Test Visualization", fontsize=22, fontweight='bold', y=0.98)

    # Add parameter information at the top
    param_ax = fig.add_subplot(gs[0, :])
    param_ax.axis('off')
    param_text = (
        f"Disease Prevalence: {prior_prob:.1%}  |  "
        f"Test Sensitivity: {sensitivity:.1%}  |  "
        f"Test Specificity: {specificity:.1%}"
    )
    param_ax.text(0.5, 0.5, param_text, ha='center', va='center', fontsize=16,
                 bbox=dict(facecolor='#e6f2ff', edgecolor='#3399ff', boxstyle='round,pad=0.5'))

    # 1. Population Distribution (Left)
    pop_ax = fig.add_subplot(gs[1, 0])
    pop_ax.set_title("Population Distribution", fontsize=18, pad=15)
    pop_ax.axis('equal')
    pop_ax.set_xlim(0, 100)
    pop_ax.set_ylim(0, 100)

    # Create a clean, modern look with a light grid
    pop_ax.grid(False)
    pop_ax.set_xticks([])
    pop_ax.set_yticks([])
    pop_ax.spines['top'].set_visible(False)
    pop_ax.spines['right'].set_visible(False)
    pop_ax.spines['bottom'].set_visible(False)
    pop_ax.spines['left'].set_visible(False)

    # Draw the population rectangle with a light border
    pop_rect = patches.Rectangle((0, 0), 100, 100, linewidth=2, edgecolor='#666666', facecolor='#f0f0f0', alpha=0.3)
    pop_ax.add_patch(pop_rect)

    # Draw the disease prevalence with a distinct color
    disease_width = prior_prob * 100
    disease_rect = patches.Rectangle((0, 0), disease_width, 100, linewidth=1,
                                    edgecolor='#cc0000', facecolor='#ff9999', alpha=0.7)
    pop_ax.add_patch(disease_rect)

    # Add clear labels with contrasting backgrounds
    pop_ax.text(disease_width/2, 50, f"Have disease\n{int(prior_prob*population)} people\n({prior_prob:.1%})",
               ha='center', va='center', fontsize=14, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='#cc0000', boxstyle='round,pad=0.3'))

    pop_ax.text(disease_width + (100-disease_width)/2, 50,
               f"Don't have disease\n{int((1-prior_prob)*population)} people\n({1-prior_prob:.1%})",
               ha='center', va='center', fontsize=14, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='#666666', boxstyle='round,pad=0.3'))

    # Add a dividing line
    pop_ax.axvline(x=disease_width, color='#666666', linestyle='--', linewidth=2, alpha=0.7)

    # 2. Test Results (Right)
    test_ax = fig.add_subplot(gs[1, 1])
    test_ax.set_title("Test Results", fontsize=18, pad=15)
    test_ax.axis('equal')
    test_ax.set_xlim(0, 100)
    test_ax.set_ylim(0, 100)

    # Clean styling
    test_ax.grid(False)
    test_ax.set_xticks([])
    test_ax.set_yticks([])
    test_ax.spines['top'].set_visible(False)
    test_ax.spines['right'].set_visible(False)
    test_ax.spines['bottom'].set_visible(False)
    test_ax.spines['left'].set_visible(False)

    # Draw the test results rectangle
    test_rect = patches.Rectangle((0, 0), 100, 100, linewidth=2, edgecolor='#666666', facecolor='#f0f0f0', alpha=0.3)
    test_ax.add_patch(test_rect)

    # Calculate proportions for visualization
    tp_width = (prior_prob * sensitivity) * 100
    fp_width = ((1 - prior_prob) * (1 - specificity)) * 100
    fn_width = (prior_prob * (1 - sensitivity)) * 100
    tn_width = ((1 - prior_prob) * specificity) * 100

    # Use a clear, distinct color palette
    tp_rect = patches.Rectangle((0, 0), tp_width, 100, linewidth=1,
                               edgecolor='#990000', facecolor='#ff5555', alpha=0.8)
    test_ax.add_patch(tp_rect)

    fp_rect = patches.Rectangle((tp_width, 0), fp_width, 100, linewidth=1,
                               edgecolor='#994400', facecolor='#ffaa77', alpha=0.8)
    test_ax.add_patch(fp_rect)

    fn_rect = patches.Rectangle((tp_width + fp_width, 0), fn_width, 100, linewidth=1,
                               edgecolor='#004499', facecolor='#77aaff', alpha=0.8)
    test_ax.add_patch(fn_rect)

    tn_rect = patches.Rectangle((tp_width + fp_width + fn_width, 0), tn_width, 100, linewidth=1,
                               edgecolor='#000066', facecolor='#5588ff', alpha=0.8)
    test_ax.add_patch(tn_rect)

    # Add dividing lines
    test_ax.axvline(x=tp_width + fp_width, color='#666666', linestyle='--', linewidth=2, alpha=0.7)
    test_ax.axvline(x=tp_width, color='#666666', linestyle=':', linewidth=1.5, alpha=0.7)
    test_ax.axvline(x=tp_width + fp_width + fn_width, color='#666666', linestyle=':', linewidth=1.5, alpha=0.7)

    # Improved label placement to avoid overlap
    # Use vertical positioning to separate labels

    # True Positives - top position
    test_ax.text(tp_width/2, 75, f"True Positives\n{int(true_positive)} people",
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='#990000', boxstyle='round,pad=0.3'))

    # False Positives - bottom position if narrow, otherwise center
    if fp_width < 10:  # If the section is narrow
        fp_y_pos = 25 if tp_width > 10 else 50  # Adjust based on TP width
        test_ax.text(tp_width + fp_width/2, fp_y_pos, f"False\nPositives\n{int(false_positive)}",
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='#994400', boxstyle='round,pad=0.3'))
    else:
        test_ax.text(tp_width + fp_width/2, 50, f"False Positives\n{int(false_positive)} people",
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='#994400', boxstyle='round,pad=0.3'))

    # False Negatives - top position if narrow, otherwise center
    if fn_width < 10:  # If the section is narrow
        fn_y_pos = 75 if tn_width > 10 else 50  # Adjust based on TN width
        test_ax.text(tp_width + fp_width + fn_width/2, fn_y_pos, f"False\nNegatives\n{int(false_negative)}",
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='#004499', boxstyle='round,pad=0.3'))
    else:
        test_ax.text(tp_width + fp_width + fn_width/2, 50, f"False Negatives\n{int(false_negative)} people",
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='#004499', boxstyle='round,pad=0.3'))

    # True Negatives - bottom position
    test_ax.text(tp_width + fp_width + fn_width + tn_width/2, 25, f"True Negatives\n{int(true_negative)} people",
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='#000066', boxstyle='round,pad=0.3'))

    # Add test positive/negative regions with clear separation and improved positioning
    test_positive = true_positive + false_positive
    test_negative = false_negative + true_negative

    # Create a separate area below the main visualization for test result summaries
    test_ax.add_patch(patches.Rectangle((0, -20), tp_width + fp_width, 15,
                                       facecolor='#ffeeee', alpha=0.7, edgecolor='#cc0000'))
    test_ax.add_patch(patches.Rectangle((tp_width + fp_width, -20), fn_width + tn_width, 15,
                                       facecolor='#eeeeff', alpha=0.7, edgecolor='#0044cc'))

    # Add clear labels for test positive/negative
    test_ax.text(tp_width/2 + fp_width/2, -12.5,
                f"Test Positive: {int(test_positive)} people ({test_positive/population:.1%})",
                ha='center', va='center', fontsize=14, fontweight='bold', color='#990000')

    test_ax.text(tp_width + fp_width + fn_width/2 + tn_width/2, -12.5,
                f"Test Negative: {int(test_negative)} people ({test_negative/population:.1%})",
                ha='center', va='center', fontsize=14, fontweight='bold', color='#000066')

    # 3. Formula and Conclusion (Bottom)
    formula_ax = fig.add_subplot(gs[2, :])
    formula_ax.axis('off')

    # Create a box for the formula
    formula_box = patches.FancyBboxPatch((0.1, 0.4), 0.8, 0.5, boxstyle=patches.BoxStyle("Round", pad=0.6),
                                        facecolor='#f5f5f5', edgecolor='#3399ff', linewidth=2, alpha=0.7,
                                        transform=formula_ax.transAxes)
    formula_ax.add_patch(formula_box)

    # Add Bayes formula with clear formatting
    formula_title = "Bayes' Theorem Applied to Medical Testing:"
    formula_ax.text(0.5, 0.8, formula_title, ha='center', va='center', fontsize=16, fontweight='bold',
                   transform=formula_ax.transAxes)

    formula = r"$P(Disease|Positive) = \frac{P(Positive|Disease) \times P(Disease)}{P(Positive)}$"
    formula_ax.text(0.5, 0.65, formula, ha='center', va='center', fontsize=16,
                   transform=formula_ax.transAxes)

    formula_explained = "Posterior Probability = Sensitivity × Prior Probability / Probability of Positive Test"
    formula_ax.text(0.5, 0.5, formula_explained, ha='center', va='center', fontsize=14, color='#555555',
                   transform=formula_ax.transAxes)

    # Add the calculation with the actual values
    test_positive_prob = test_positive/population
    calculation = f"= {sensitivity:.1%} × {prior_prob:.1%} / {test_positive_prob:.1%} = {posterior_prob:.1%}"
    formula_ax.text(0.5, 0.35, calculation, ha='center', va='center', fontsize=16,
                   transform=formula_ax.transAxes)

    # Create a highlighted conclusion box
    conclusion_box = patches.FancyBboxPatch((0.15, 0.05), 0.7, 0.2, boxstyle=patches.BoxStyle("Round", pad=0.6),
                                          facecolor='#ffffcc', edgecolor='#ffcc00', linewidth=2,
                                          transform=formula_ax.transAxes)
    formula_ax.add_patch(conclusion_box)

    # Add the conclusion with emphasis
    conclusion = f"If someone tests positive, they have a {posterior_prob:.1%} chance of having the disease"
    formula_ax.text(0.5, 0.15, conclusion, ha='center', va='center', fontsize=18, fontweight='bold',
                   transform=formula_ax.transAxes)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.1, wspace=0.1)

    return fig

def explain_bayes(prior_prob, sensitivity, specificity):
    """Generate the Bayes' theorem explanation and visualization."""
    population = 1000

    # Calculate values based on Bayes theorem
    true_positive = prior_prob * sensitivity * population
    false_positive = (1 - prior_prob) * (1 - specificity) * population
    true_negative = (1 - prior_prob) * specificity * population
    false_negative = prior_prob * (1 - sensitivity) * population

    # Calculate posterior probability (positive predictive value)
    test_positive = true_positive + false_positive
    test_positive_prob = test_positive / population
    posterior_prob = true_positive / test_positive if test_positive > 0 else 0

    # Create the visualization
    fig = create_bayes_visualization(prior_prob, sensitivity, specificity, population)

    # Generate explanation text with clearer explanations of the percentages
    explanation = f"""
### Medical Test Example Explained Step-by-Step

Imagine a medical test for a disease that affects {prior_prob:.1%} of the population (prior probability).

**What the percentages mean:**

1. **Disease Prevalence ({prior_prob:.1%})**:
   - This means that out of every 100 people, about {int(prior_prob*100)} people have the disease
   - In our population of 1,000 people, {int(prior_prob*population)} people have the disease and {int((1-prior_prob)*population)} people don't

2. **Test Sensitivity ({sensitivity:.1%})**:
   - This means the test correctly identifies {sensitivity:.1%} of people who actually have the disease
   - Out of the {int(prior_prob*population)} people with the disease:
     * {int(true_positive)} people test positive (true positives) = {int(prior_prob*population)} × {sensitivity:.1%}
     * {int(false_negative)} people test negative (false negatives) = {int(prior_prob*population)} × {(1-sensitivity):.1%}

3. **Test Specificity ({specificity:.1%})**:
   - This means the test correctly identifies {specificity:.1%} of people who don't have the disease
   - Out of the {int((1-prior_prob)*population)} people without the disease:
     * {int(true_negative)} people test negative (true negatives) = {int((1-prior_prob)*population)} × {specificity:.1%}
     * {int(false_positive)} people test positive (false positives) = {int((1-prior_prob)*population)} × {(1-specificity):.1%}

4. **Probability of Positive Test ({test_positive_prob:.1%})**:
   - This is the total percentage of people who test positive, regardless of whether they have the disease
   - It's calculated by adding:
     * True positives: {int(true_positive)} people = {prior_prob:.1%} × {sensitivity:.1%} × 1,000
     * False positives: {int(false_positive)} people = {(1-prior_prob):.1%} × {(1-specificity):.1%} × 1,000
   - Total positive tests: {int(test_positive)} people out of 1,000 = {test_positive_prob:.1%} of the population

**How the formula works:**

Bayes' theorem calculates the probability that someone actually has the disease if they test positive:

P(Disease|Positive) = P(Positive|Disease) × P(Disease) / P(Positive)

Breaking this down with our numbers:
- P(Positive|Disease) = Sensitivity = {sensitivity:.1%}
- P(Disease) = Prior Probability = {prior_prob:.1%}
- P(Positive) = Probability of a positive test = {test_positive_prob:.1%}

Putting these into the formula:
- Posterior Probability = {sensitivity:.1%} × {prior_prob:.1%} ÷ {test_positive_prob:.1%}
- = {sensitivity * prior_prob:.1%} ÷ {test_positive_prob:.1%}
- = {posterior_prob:.1%}

**The key insight:** If someone tests positive, they have a {posterior_prob:.1%} chance of having the disease, not {sensitivity:.1%} as many people might think!

This is often surprising because:
1. Even a good test ({sensitivity:.1%} accurate) can give misleading results when a disease is rare
2. Most positive results might actually be false alarms when testing for rare conditions
3. The more common a disease is, the more likely a positive test is to be correct
"""

    return fig, explanation

# Create the Gradio interface
with gr.Blocks(title="Bayes' Theorem Visualizer") as demo:
    gr.Markdown("# Bayes' Theorem Visualizer")
    gr.Markdown("""
    Bayes' theorem helps us update our beliefs based on new evidence. This interactive tool visualizes how prior probability,
    sensitivity, and specificity affect the posterior probability in a medical testing scenario.

    Adjust the sliders below and see how the results change in real-time!
    """)

    with gr.Column():
        with gr.Group():
            gr.Markdown("### Adjust Parameters")

            prior_prob = gr.Slider(
                minimum=0.01, maximum=0.5, value=0.1, step=0.01,
                label="Disease Prevalence (Prior Probability)",
                info="What percentage of the population has the disease?"
            )

            sensitivity = gr.Slider(
                minimum=0.5, maximum=1.0, value=0.9, step=0.01,
                label="Test Sensitivity (True Positive Rate)",
                info="How good is the test at detecting people who have the disease?"
            )

            specificity = gr.Slider(
                minimum=0.5, maximum=1.0, value=0.9, step=0.01,
                label="Test Specificity (True Negative Rate)",
                info="How good is the test at correctly identifying people who don't have the disease?"
            )

        output_plot = gr.Plot(label="Visualization")
        output_text = gr.Markdown(label="Explanation")

        with gr.Accordion("Key Terms", open=False):
            gr.Markdown("""
            - **Prior Probability (Prevalence)**: The initial probability of having a disease before testing
            - **Sensitivity**: The ability to correctly identify those with the disease (true positive rate)
            - **Specificity**: The ability to correctly identify those without the disease (true negative rate)
            - **Posterior Probability**: The updated probability of having the disease after a positive test
            - **True Positive**: Correctly identified as having the disease
            - **False Positive**: Incorrectly identified as having the disease (also called a "Type I error")
            - **True Negative**: Correctly identified as not having the disease
            - **False Negative**: Incorrectly identified as not having the disease (also called a "Type II error")
            """)

    # Update when any parameter changes
    for param in [prior_prob, sensitivity, specificity]:
        param.change(
            explain_bayes,
            inputs=[prior_prob, sensitivity, specificity],
            outputs=[output_plot, output_text]
        )

    # Add examples
    gr.Examples(
        examples=[
            [0.01, 0.99, 0.99],  # Rare disease, excellent test
            [0.1, 0.9, 0.9],     # Common scenario
            [0.3, 0.8, 0.7],     # More common disease, less accurate test
            [0.5, 0.7, 0.95]     # Very common disease, asymmetric test accuracy
        ],
        inputs=[prior_prob, sensitivity, specificity],
        outputs=[output_plot, output_text],
        fn=explain_bayes,
        label="Try These Examples"
    )

    # Initialize the visualization
    demo.load(
        explain_bayes,
        inputs=[prior_prob, sensitivity, specificity],
        outputs=[output_plot, output_text]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
