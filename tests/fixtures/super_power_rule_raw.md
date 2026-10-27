

**Core idea:** The power rule states that the derivative of \(x\) raised to any real exponent \(n\) is \(n\) multiplied by \(x\) to the power \((n-1)\).  

## Statement of the Power Rule  
For any real number \(n\),  
\[
\frac{d}{dx}\,x^{n}=n\,x^{\,n-1}
\]  
This formula holds for positive, negative, fractional, and irrational exponents \([3][4][5]\).

## How to Apply the Rule  
1. **Identify the exponent** \(n\) on the base \(x\).  
2. **Subtract 1** from that exponent.  
3. **Multiply** the result by the original exponent \(n\).  

*Example:* For \(f(x)=x^{5}\),  
- exponent \(n=5\)  
- \(n-1 = 4\)  
- derivative \(f'(x)=5x^{4}\) \([3]\).

## Proof Sketch for Positive Integer Powers  
When \(n\) is a positive integer, the difference of powers factors as  
\[
z^{n}-x^{n}=(z-x)\bigl(z^{n-1}+z^{n-2}x+\dots+zx^{n-2}+x^{n-1}\bigr).
\]  
Dividing by \(z-x\) and letting \(z\to x\) leaves \(n\) terms, each equal to \(x^{n-1}\); thus the limit is \(nx^{n-1}\) \([4]\).

## Extension to All Real Exponents  
The definition of the general exponential function allows us to write \(y=x^{n}\) for any real \(n\) (rational or irrational). Using implicit differentiation or the limit definition shows the same formula \(nx^{n-1}\) remains valid \([2][1]\).

### Exam Contrast: Power Rule vs. Constant Multiple Rule  
| Situation                              | Rule to Use                     | Result                               |
|----------------------------------------|--------------------------------|--------------------------------------|
| Derivative of \(c\cdot x^{n}\) (c constant) | Constant Multiple + Power Rule | \(c\cdot n x^{\,n-1}\)                |
| Derivative of \(x^{n}\) alone          | Power Rule only                | \(n x^{\,n-1}\)                      |
| Derivative of a constant \(c\)         | Constant Rule                  | \(0\)                                |

A common mistake is to forget to keep the constant \(c\) when applying the power rule; the constant multiple rule tells us the constant stays as a factor \([5]\).

## Check-in  
Are you studying for a calculus course (e.g., Calculus I) and would you like a quick quiz, flashcards, or a deeper breakdown of any differentiation rule?