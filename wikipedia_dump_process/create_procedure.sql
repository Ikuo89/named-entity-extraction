DROP TABLE IF EXISTS amusements;
CREATE TABLE `amusements` (
  `id` int(8) unsigned NOT NULL AUTO_INCREMENT,
  `title` varbinary(255) NOT NULL DEFAULT '',
  `sub_type` enum('page','subcat','file') NOT NULL DEFAULT 'page',
  `last_checked_at` datetime NOT NULL DEFAULT '1900-01-01',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_title` (`title`, `sub_type`),
  KEY `index_batch` (`last_checked_at`, `sub_type`)
) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=binary;
INSERT INTO amusements (title, sub_type) VALUES ('娯楽', 'subcat');

DROP PROCEDURE IF EXISTS exclude_wikipedia_hobbies;
DELIMITER //
CREATE PROCEDURE exclude_wikipedia_hobbies()
begin
  DECLARE target_cat varbinary(255);

  read_loop: LOOP
    SELECT title INTO target_cat FROM amusements WHERE last_checked_at < (now() - interval 1 day) AND sub_type = 'subcat' ORDER BY last_checked_at ASC LIMIT 1;

    IF target_cat IS NULL THEN
      LEAVE read_loop;
    END IF;

    INSERT IGNORE INTO amusements (title, sub_type)
      select
        page.page_title,
        cl.cl_type
      from
        categorylinks as cl
        left join page on page.page_id = cl.cl_from
      where
        cl.cl_to = target_cat;
    UPDATE amusements SET last_checked_at = now() WHERE title = target_cat;
  END LOOP;
end
//
DELIMITER ;
CALL exclude_wikipedia_hobbies;
