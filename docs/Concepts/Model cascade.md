# Model cascade

The standard way to make a semantic filter affordable. A cheap *proxy* model scores every row; only rows whose score is uncertain are escalated to the expensive *oracle* model. Thresholds are calibrated so the final result meets a user-specified accuracy target. The saving comes from rows the proxy can settle on its own.
