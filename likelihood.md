# Metropolis Random Walk with spatially variable observational uncertainty

$$ L(m) = k \exp{\left[ \frac{-1}{2} \sum \frac{\left( g_i - d_i \right)^2}{\sigma_i^2} \right]} $$

$$ \alpha = \min \left[ 1, \frac{L(m_{\text{new}})}{L(m_{\text{old}})} \right]  = \exp \left[ \frac{-1}{2}\sum \frac{\left(g_{\text{new}}^i - d^i \right)^2}{\sigma^{i^2}} + \frac{1}{2}\sum \frac{\left(g_{\text{old}}^i - d^i \right)^2}{\sigma^{i^2}} \right] $$