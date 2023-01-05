library(dplyr)
library(ggplot2)

imp <- read.csv('var_imp.csv', row.names = 1)
# imp <- imp %>% top_n(10)
imp <- imp %>% arrange(desc(Overall))
names <- rownames(imp)

ggplot(imp, aes(x=factor(names, level=rev(names)), y=Overall)) +
  geom_point( color="blue", size=4, alpha=0.6)+
  geom_segment( aes(x=names, xend=names, y=0, yend=Overall), 
                color='black') +
  xlab('Feature')+
  ylab('Overall Importance')+
  theme_light() +
  coord_flip() 