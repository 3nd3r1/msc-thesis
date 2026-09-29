# Model cascade

The usual way to make a semantic filter affordable. A cheap proxy model scores every row, and the rows it can settle on its own never reach the expensive oracle model. Only the uncertain ones are escalated. Thresholds are calibrated so the final result still meets a user-given accuracy target.
