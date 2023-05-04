write.mts <- function(ts, filename) {
  t = cbind(Year=time(ts), ts)
  colnames(t) <- c("Year", colnames(ts))
  write.csv(t, filename)
}