"""
Statistical analysis utilities for HAD-MC 2.0 experiments

This module provides:
- Paired t-tests
- Wilcoxon signed-rank tests
- ANOVA for multi-group comparison
- Bootstrap confidence intervals
- Effect size calculations (Cohen's d)
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional
import warnings


def paired_t_test(
    group1: List[float],
    group2: List[float],
    alpha: float = 0.05
) -> Dict:
    """
    Perform paired t-test to compare two groups.

    Args:
        group1: First group of values
        group2: Second group of values (same length as group1)
        alpha: Significance level

    Returns:
        Dictionary containing:
            - t_statistic: t-statistic value
            - p_value: p-value
            - significant: Whether difference is significant at alpha
            - mean_diff: Mean difference (group1 - group2)
            - ci_lower: Lower bound of confidence interval
            - ci_upper: Upper bound of confidence interval
    """
    if len(group1) != len(group2):
        raise ValueError("Groups must have same length for paired t-test")

    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(group1, group2)

    # Calculate mean difference
    mean_diff = np.mean(np.array(group1) - np.array(group2))

    # Calculate confidence interval
    se = stats.sem(np.array(group1) - np.array(group2))
    ci = stats.t.interval(1 - alpha, len(group1) - 1, loc=mean_diff, scale=se)

    return {
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'mean_diff': float(mean_diff),
        'ci_lower': float(ci[0]),
        'ci_upper': float(ci[1]),
        'alpha': alpha
    }


def wilcoxon_signed_rank_test(
    group1: List[float],
    group2: List[float],
    alpha: float = 0.05
) -> Dict:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Args:
        group1: First group of values
        group2: Second group of values
        alpha: Significance level

    Returns:
        Dictionary containing:
            - statistic: Wilcoxon statistic
            - p_value: p-value
            - significant: Whether difference is significant at alpha
            - median_diff: Median difference
    """
    if len(group1) != len(group2):
        raise ValueError("Groups must have same length for Wilcoxon test")

    # Perform Wilcoxon signed-rank test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, p_value = stats.wilcoxon(group1, group2)

    # Calculate median difference
    median_diff = np.median(np.array(group1) - np.array(group2))

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'median_diff': float(median_diff),
        'alpha': alpha
    }


def anova_test(
    *groups: List[float],
    alpha: float = 0.05
) -> Dict:
    """
    Perform one-way ANOVA to compare multiple groups.

    Args:
        *groups: Variable number of groups to compare
        alpha: Significance level

    Returns:
        Dictionary containing:
            - f_statistic: F-statistic
            - p_value: p-value
            - significant: Whether there are significant differences
            - df_between: Degrees of freedom between groups
            - df_within: Degrees of freedom within groups
    """
    if len(groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups")

    # Perform one-way ANOVA
    f_statistic, p_value = stats.f_oneway(*groups)

    # Degrees of freedom
    k = len(groups)
    n = sum(len(group) for group in groups)
    df_between = k - 1
    df_within = n - k

    return {
        'f_statistic': float(f_statistic),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'df_between': df_between,
        'df_within': df_within,
        'alpha': alpha
    }


def bootstrap_ci(
    data: List[float],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    statistic: str = 'mean'
) -> Dict:
    """
    Calculate bootstrap confidence interval.

    Args:
        data: List of values
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence interval level
        statistic: Statistic to compute ('mean', 'median', 'std')

    Returns:
        Dictionary containing:
            - estimate: Observed statistic value
            - ci_lower: Lower bound of CI
            - ci_upper: Upper bound of CI
            - bootstrap_values: All bootstrap sample values
    """
    data = np.array(data)
    n = len(data)

    # Calculate observed statistic
    if statistic == 'mean':
        estimate = np.mean(data)
    elif statistic == 'median':
        estimate = np.median(data)
    elif statistic == 'std':
        estimate = np.std(data, ddof=1)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Generate bootstrap samples
    bootstrap_values = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)

        if statistic == 'mean':
            bootstrap_values.append(np.mean(bootstrap_sample))
        elif statistic == 'median':
            bootstrap_values.append(np.median(bootstrap_sample))
        elif statistic == 'std':
            bootstrap_values.append(np.std(bootstrap_sample, ddof=1))

    bootstrap_values = np.array(bootstrap_values)

    # Calculate confidence interval
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_values, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)

    return {
        'estimate': float(estimate),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'bootstrap_values': bootstrap_values.tolist(),
        'ci_level': ci_level
    }


def cohens_d(
    group1: List[float],
    group2: List[float]
) -> Dict:
    """
    Calculate Cohen's d effect size.

    Cohen's d measures the standardized difference between two means.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Dictionary containing:
            - cohens_d: Cohen's d value
            - interpretation: Interpretation of effect size magnitude
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    # Calculate pooled standard deviation
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Calculate Cohen's d
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std

    # Interpret effect size
    if abs(cohens_d) < 0.2:
        interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        interpretation = "small"
    elif abs(cohens_d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        'cohens_d': float(cohens_d),
        'interpretation': interpretation
    }


def multiple_comparison_correction(
    p_values: List[float],
    method: str = 'bonferroni'
) -> List[float]:
    """
    Apply multiple comparison correction to p-values.

    Args:
        p_values: List of p-values
        method: Correction method ('bonferroni', 'fdr_bh', 'holm')

    Returns:
        List of corrected p-values
    """
    p_values = np.array(p_values)
    n = len(p_values)

    if method == 'bonferroni':
        # Bonferroni correction
        corrected = p_values * n
        corrected = np.minimum(corrected, 1.0)
    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR correction
        from scipy.stats import multitest
        _, corrected, _, _ = multitest.multipletests(
            p_values,
            alpha=0.05,
            method='fdr_bh'
        )
    elif method == 'holm':
        # Holm-Bonferroni correction
        from scipy.stats import multitest
        _, corrected, _, _ = multitest.multipletests(
            p_values,
            alpha=0.05,
            method='holm'
        )
    else:
        raise ValueError(f"Unknown correction method: {method}")

    return corrected.tolist()


def sign_test(
    group1: List[float],
    group2: List[float],
    alpha: float = 0.05
) -> Dict:
    """
    Perform sign test (non-parametric test for paired data).

    Args:
        group1: First group of values
        group2: Second group of values
        alpha: Significance level

    Returns:
        Dictionary containing:
            - statistic: Number of positive differences
            - n: Total number of non-zero differences
            - p_value: p-value
            - significant: Whether difference is significant
    """
    if len(group1) != len(group2):
        raise ValueError("Groups must have same length for sign test")

    group1 = np.array(group1)
    group2 = np.array(group2)

    # Calculate differences
    differences = group1 - group2

    # Remove zero differences
    non_zero_diffs = differences[differences != 0]
    n = len(non_zero_diffs)

    if n == 0:
        return {
            'statistic': 0,
            'n': 0,
            'p_value': 1.0,
            'significant': False
        }

    # Count positive signs
    statistic = np.sum(non_zero_diffs > 0)

    # Perform binomial test
    p_value = stats.binom_test(statistic, n=n, p=0.5, alternative='two-sided')

    return {
        'statistic': int(statistic),
        'n': n,
        'p_value': float(p_value),
        'significant': p_value < alpha
    }


def permutation_test(
    group1: List[float],
    group2: List[float],
    n_permutations: int = 10000,
    statistic: str = 'mean'
) -> Dict:
    """
    Perform permutation test (non-parametric test for comparing groups).

    Args:
        group1: First group of values
        group2: Second group of values
        n_permutations: Number of permutations
        statistic: Statistic to use ('mean', 'median')

    Returns:
        Dictionary containing:
            - observed: Observed test statistic
            - p_value: p-value
            - significant: Whether difference is significant at alpha=0.05
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    combined = np.concatenate([group1, group2])

    # Calculate observed statistic
    if statistic == 'mean':
        observed = np.mean(group1) - np.mean(group2)
    elif statistic == 'median':
        observed = np.median(group1) - np.median(group2)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    n1 = len(group1)
    n_total = len(combined)

    # Permutation test
    perm_stats = []
    for _ in range(n_permutations):
        permuted = np.random.permutation(combined)
        perm_group1 = permuted[:n1]
        perm_group2 = permuted[n1:]

        if statistic == 'mean':
            perm_stat = np.mean(perm_group1) - np.mean(perm_group2)
        else:
            perm_stat = np.median(perm_group1) - np.median(perm_group2)

        perm_stats.append(perm_stat)

    perm_stats = np.array(perm_stats)

    # Calculate p-value (two-tailed)
    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))

    return {
        'observed': float(observed),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'n_permutations': n_permutations
    }


def summarize_statistics(
    data: List[float],
    include_bootstrap: bool = True,
    n_bootstrap: int = 10000
) -> Dict:
    """
    Summarize statistics for a single group.

    Args:
        data: List of values
        include_bootstrap: Whether to include bootstrap CI
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary of statistics
    """
    data = np.array(data)

    result = {
        'n': len(data),
        'mean': float(np.mean(data)),
        'std': float(np.std(data, ddof=1)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75))
    }

    # Calculate SEM
    result['sem'] = result['std'] / np.sqrt(result['n'])

    # Add bootstrap CI
    if include_bootstrap and len(data) > 1:
        ci = bootstrap_ci(data, n_bootstrap=n_bootstrap, ci_level=0.95)
        result['ci_lower'] = ci['ci_lower']
        result['ci_upper'] = ci['ci_upper']
        result['ci_level'] = 0.95

    return result


def compare_groups(
    groups: Dict[str, List[float]],
    alpha: float = 0.05
) -> Dict:
    """
    Compare multiple groups using appropriate statistical tests.

    Args:
        groups: Dictionary mapping group names to values
        alpha: Significance level

    Returns:
        Dictionary containing comparison results
    """
    group_names = list(groups.keys())
    group_values = [groups[name] for name in group_names]

    result = {
        'groups': group_names,
        'summary': {}
    }

    # Summary statistics for each group
    for name, values in groups.items():
        result['summary'][name] = summarize_statistics(values)

    # Pairwise comparisons
    result['pairwise'] = {}
    p_values = []

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            name1, name2 = group_names[i], group_names[j]
            values1, values2 = group_values[i], group_values[j]

            # Perform paired t-test
            t_test = paired_t_test(values1, values2, alpha=alpha)
            p_values.append(t_test['p_value'])

            result['pairwise'][f'{name1}_vs_{name2}'] = t_test

    # Multiple comparison correction
    if p_values:
        corrected = multiple_comparison_correction(p_values, method='bonferroni')
        idx = 0
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                name1, name2 = group_names[i], group_names[j]
                key = f'{name1}_vs_{name2}'
                result['pairwise'][key]['p_value_corrected'] = float(corrected[idx])
                result['pairwise'][key]['significant_corrected'] = corrected[idx] < alpha
                idx += 1

    # ANOVA if more than 2 groups
    if len(groups) > 2:
        anova = anova_test(*group_values, alpha=alpha)
        result['anova'] = anova

    return result
