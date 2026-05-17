# Figure Captions and Interpretation

## agreement_by_material_and_model.png

- Figure file: `/Users/askarovi/Documents/New project/figures/model_comparison/agreement_by_material_and_model.png`
- LaTeX:
```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/model_comparison/agreement_by_material_and_model.png}
  \caption{The plot compares how far each model deviates from the physics-informed PINN baseline under identical 2D source-probe conditions for each rock type.}
  \label{fig:agreement_by_material_and_model}
\end{figure}
```
- Interpretation: Agreement Deviation by Material and Model.
- Warnings / limitations: These deviations indicate comparative prototype behavior rather than physical error against ground truth.

## pairwise_model_difference_heatmap.png

- Figure file: `/Users/askarovi/Documents/New project/figures/model_comparison/pairwise_model_difference_heatmap.png`
- LaTeX:
```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/model_comparison/pairwise_model_difference_heatmap.png}
  \caption{The heatmap summarizes pairwise differences in predicted maximum displacement between model services across the analyzed 2D cases.}
  \label{fig:pairwise_model_difference_heatmap}
\end{figure}
```
- Interpretation: Pairwise Model Difference Heatmap.
- Warnings / limitations: Large values, especially when dominated by FNO, should be interpreted as scale instability or model disagreement rather than proof of real physical divergence.

## error_or_deviation_vs_density.png

- Figure file: `/Users/askarovi/Documents/New project/figures/model_comparison/error_or_deviation_vs_density.png`
- LaTeX:
```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/model_comparison/error_or_deviation_vs_density.png}
  \caption{The plot shows how the model deviation score changes with material density across the available 2D experiments.}
  \label{fig:error_or_deviation_vs_density}
\end{figure}
```
- Interpretation: Deviation vs Density.
- Warnings / limitations: The trend suggests an observed association only; it should not be interpreted as field-validated proof of density-controlled wave behavior.

## error_or_deviation_vs_young_modulus.png

- Figure file: `/Users/askarovi/Documents/New project/figures/model_comparison/error_or_deviation_vs_young_modulus.png`
- LaTeX:
```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/model_comparison/error_or_deviation_vs_young_modulus.png}
  \caption{The plot compares deviation from the PINN baseline as a function of estimated Young's modulus for the analyzed rocks.}
  \label{fig:error_or_deviation_vs_young_modulus}
\end{figure}
```
- Interpretation: Deviation vs Young's Modulus.
- Warnings / limitations: The relationship is qualitative and reflects comparative model behavior within the prototype setup.

## error_or_deviation_vs_thermal_conductivity.png

- Figure file: `/Users/askarovi/Documents/New project/figures/model_comparison/error_or_deviation_vs_thermal_conductivity.png`
- LaTeX:
```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/model_comparison/error_or_deviation_vs_thermal_conductivity.png}
  \caption{The plot compares deviation from the PINN baseline against thermal conductivity for the available geological media.}
  \label{fig:error_or_deviation_vs_thermal_conductivity}
\end{figure}
```
- Interpretation: Deviation vs Thermal Conductivity.
- Warnings / limitations: This should be treated as a prototype-level observed association, not a validated thermoelastic law.

## directional_error_or_deviation_by_azimuth.png

- Figure file: `/Users/askarovi/Documents/New project/figures/model_comparison/directional_error_or_deviation_by_azimuth.png`
- LaTeX:
```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/model_comparison/directional_error_or_deviation_by_azimuth.png}
  \caption{The plot evaluates whether model disagreement changes with the imposed 2D source-probe direction.}
  \label{fig:directional_error_or_deviation_by_azimuth}
\end{figure}
```
- Interpretation: Directional Deviation by Input Azimuth.
- Warnings / limitations: The graph is valid only because the input azimuth is computed from shared source-probe geometry and is therefore consistent across models for each case.

## outlier_count_by_model.png

- Figure file: `/Users/askarovi/Documents/New project/figures/model_comparison/outlier_count_by_model.png`
- LaTeX:
```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/model_comparison/outlier_count_by_model.png}
  \caption{The plot summarizes prototype-level numerical warning counts for each model service.}
  \label{fig:outlier_count_by_model}
\end{figure}
```
- Interpretation: Outlier Count by Model.
- Warnings / limitations: A higher count indicates lower numerical stability or scale consistency in the current implementation, not necessarily a physically impossible response.

## feature_sensitivity_heatmap.png

- Figure file: `/Users/askarovi/Documents/New project/figures/model_comparison/feature_sensitivity_heatmap.png`
- LaTeX:
```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/model_comparison/feature_sensitivity_heatmap.png}
  \caption{The heatmap summarizes how strongly each model output varies with physical material parameters across the available 2D experiments.}
  \label{fig:feature_sensitivity_heatmap}
\end{figure}
```
- Interpretation: Observed Feature-Output Association Heatmap.
- Warnings / limitations: Because no controlled perturbation study was run here, this figure shows observed association rather than controlled sensitivity.

## max_displacement_without_fno.png

- Figure file: `/Users/askarovi/Documents/New project/figures/model_comparison/max_displacement_without_fno.png`
- LaTeX:
```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/model_comparison/max_displacement_without_fno.png}
  \caption{The plot compares non-FNO displacement magnitudes after excluding the scale-unstable FNO baseline for readability.}
  \label{fig:max_displacement_without_fno}
\end{figure}
```
- Interpretation: Maximum Displacement without FNO.
- Warnings / limitations: It is included as a diagnostic aid and should be interpreted together with the explicit FNO diagnostic report.

## temperature_perturbation_without_fno.png

- Figure file: `/Users/askarovi/Documents/New project/figures/model_comparison/temperature_perturbation_without_fno.png`
- LaTeX:
```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/model_comparison/temperature_perturbation_without_fno.png}
  \caption{The plot compares non-FNO temperature perturbation predictions after excluding the scale-unstable FNO baseline.}
  \label{fig:temperature_perturbation_without_fno}
\end{figure}
```
- Interpretation: Temperature Perturbation without FNO.
- Warnings / limitations: This improves readability but does not remove the need to discuss FNO separately as an unstable prototype baseline.

## fno_scale_outlier_diagnostic.png

- Figure file: `/Users/askarovi/Documents/New project/figures/model_comparison/fno_scale_outlier_diagnostic.png`
- LaTeX:
```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/model_comparison/fno_scale_outlier_diagnostic.png}
  \caption{The plot visualizes the difference in displacement scale between FNO and the other model services.}
  \label{fig:fno_scale_outlier_diagnostic}
\end{figure}
```
- Interpretation: FNO Scale Outlier Diagnostic.
- Warnings / limitations: This diagnostic supports the interpretation that FNO remains numerically unstable in the current prototype and should not be overinterpreted physically.

- Speed-specific figures were skipped because no latency column was available in the current summary dataset.
- Accuracy-specific figures were skipped because no explicit ground-truth columns were available; agreement and stability analysis were used instead.
