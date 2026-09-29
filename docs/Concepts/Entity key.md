# Entity key

A column already in the schema that groups rows belonging to the same real-world thing: product ID for reviews, author ID for posts, thread ID for messages. Grouping by it is exact and costs nothing, while embedding clusters have to be inferred and can put the wrong rows together.
