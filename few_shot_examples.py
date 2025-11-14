few_shot_examples = [
    {
        "natural_language": "How many products are there?",
        "sql_query": """
        SELECT COUNT(DISTINCT SAP) FROM products;
        """        
    },
    {
        "natural_language": "How many units of product 1234 do we have in stock?",
        "sql_query": """
        SELECT item_code, item_num, description, sum(units) FROM inventory where item_code = '1234' and date =(select max(date) from inventory) group by item_code;
        """        
    },
    {
        "natural_language": "where are the product 1234 in stock",
        "sql_query": """
        SELECT item_code, item_num, description, units, bin, exp_date FROM inventory WHERE item_code ='1234' and date = (select max(date) from inventory);
        """        
    },
    {
        "natural_language": "what is the product 1234",
        "sql_query": """
        SELECT item_code, item_num, description FROM inventory WHERE item_code = '1234' and date = (select max(date) from inventory);
        """        
    },
    {
        "natural_language": "what products are in bin 12-X-1?",
        "sql_query": """
        SELECT item_code, item_num, description units, bin, exp_date FROM inventory WHERE bin = '12-X-1' and date = (select max(date) from inventory);
        """        
    },
        {
        "natural_language": "what products are in scrap?",
        "sql_query": """
        SELECT item_code, item_num, description units, bin, exp_date FROM inventory WHERE bin = 'SCRAP' and date = (select max(date) from inventory);
        """        
    },
            {
        "natural_language": "what products are in PCNLOC?",
        "sql_query": """
        SELECT item_code, item_num, description units, bin, exp_date FROM inventory WHERE bin = 'PCNLOC' and date = (select max(date) from inventory);
        """        
    },
        {
        "natural_language": "what did we receive today?",
        "sql_query": """
        SELECT date, sap_code, item_num, qty, vendor, ref_num, location from inbound where date(date) = curdate() order by sap_code, vendor;
        """        
    },
    {
        "natural_language": "what did we receive from FNS today?",
        "sql_query": """
        SELECT date, sap_code, item_num, qty, vendor, ref_num, location from inbound where date(date) = curdate() and vendor = 'FNS';
        """        
    },
    {
        "natural_language": "how many containers did we receive in July 2025?",
        "sql_query": """
        SELECT  vendor, count(distinct(ref_num)) from inbound where date(date)>= '2025-07-01' and date(date) < '2025-08-01' and vendor = 'CONTAINER';
        """        
    },
    {
        "natural_language": "did we receive any returns today?",
        "sql_query": """
        SELECT date, sap_code, item_num, qty, vendor, ref_num , location from inbound where date(date) = curdate() and vendor = 'RETURN';
        """        
    },
    {
        "natural_language": "did we receive any containers today?",
        "sql_query": """
        SELECT date, sap_code, item_num, qty, vendor, ref_num , location from inbound where date(date) = curdate() and vendor = 'CONTAINER';
        """        
    },    
    {
        "natural_language": "show pick ticket details of product 1234",
        "sql_query": """ 
        SELECT p.status, p.date, p.so_num, s.cust_po_num, s.customer, p.sap_code, p.item_num, p.qty, p.location FROM picktix p join salesorder s on p.so_num = s.so_num and p.sap_code = s.sap_code WHERE p.sap_code = '1234' ;
        """        
    },
    {
        "natural_language": "where was the product 1234 picked from, for po 56789",
        "sql_query": """
        SELECT p.sap_code, s.cust_po_num, p.item_num, p.qty, p.location  from picktix p join salesorder s on p.so_num = s.so_num and p.sap_code = s.sap_code where p.sap_code = '1234' and s.cust_po_num = '56789';
        """        
    },
    {
        "natural_language": "show picking activity at 21-X-1",
        "sql_query": """ 
        SELECT p.status, p.date, p.so_num, s.cust_po_num, s.customer, p.sap_code, p.item_num, p.qty, p.location FROM picktix p join salesorder s on p.so_num = s.so_num and p.sap_code = s.sap_code WHERE p.location = '21-X-1' ;
        """        
    },
    {
        "natural_language": "How many containers did we unload so far?",
        "sql_query": """
        SELECT count(distinct(ref_num)) from inbound where vendor = 'CONTAINER';
        """        
    },
    {
        "natural_language": "How many containers did we unload in February, 2025?",
        "sql_query": """
        SELECT count(distinct(ref_num)) from inbound where vendor = 'CONTAINER' and date(date)>='2025-02-01' and date(date) < '2025-03-01';
        """        
    },
    {
        "natural_language": "show total number of boxes that we have received, total number of boxes that we have shipped out, difference between two and total number of boxes that warehouse finished, in February 2025",
        "sql_query": """
        SELECT (SELECT sum(ctn_num) FROM inbound WHERE date(date)>='2025-02-01' and date(date) <'2025-03-01') AS received_boxes,
       (SELECT sum(ctn_num) FROM outbound WHERE date(shipped_date)>='2025-02-01' and date(shipped_date) <'2025-03-01') AS shipped_boxes,
       ((SELECT sum(ctn_num) FROM inbound WHERE date(date)>='2025-02-01' and date(date) <'2025-03-01')) - 
       ((SELECT sum(ctn_num) FROM outbound WHERE date(shipped_date)>='2025-02-01' and date(shipped_date) <'2025-03-01')) AS difference,
       (SELECT sum(ctn_num) FROM warehouse WHERE date(process_end)>='2025-02-01' and date(process_end) <'2025-03-01') AS finished_boxes;
        """        
    },
    {
        "natural_language": "what item has upc code of 123456789?",
        "sql_query": """ 
        SELECT item_num, SAP,MASTER_QTY FROM products where UPC_M = '123456789' or UPC_U = '123456789';
        """        
    },
    {
        "natural_language": "do we have 584512091 in stock?",
        "sql_query": """
        SELECT item_code, item_num, bin, description, units, exp_date, assigned FROM inventory where item_code = '584512091';
        """        
    },
    {
        "natural_language": "where are the empty racks?",
        "sql_query": """
        SELECT bin FROM inventory GROUP BY bin HAVING SUM(units) < 1;
        """        
    },
    {
        "natural_language": "what is in SETLOC?",
        "sql_query": """
        SELECT item_code, item_num, units, bin, exp_date FROM inventory where bin like '%SETLOC%';
        """        
    },
    {
        "natural_language": "what items are assigned?",
        "sql_query": """
        SELECT item_code, item_num, units, bin, exp_date, assigned, date FROM inventory where assigned > 0;
        """        
    },
    {
        "natural_language": "is 584512091 assigned?",
        "sql_query": """
        SELECT item_code, item_num, bin, units, exp_date, assigned date FROM inventory where item_code ='584512091' and assigned > 0;
        """        
    },
    {
        "natural_language": "where was pallet 24110800033 from?",
        "sql_query": """
        SELECT sap_code, item_num, ctn_num, units,pallet_num, vendor, date FROM inbound where pallet_num like '%24110800033%' and (ctn_num> 0 or units > 0);
        """        
    },
    {
        "natural_language": "what items will expire soon?",
        "sql_query": """
        SELECT item_code, item_num, description, bin, units, exp_date FROM inventory where units > 0 and exp_date >= curdate() order by exp_date asc limit 20;
        """        
    },
    {
        "natural_language": "how many boxes of 584510753 did we receive on 2024-11-11?",
        "sql_query": """
        SELECT sap_code, item_num, sum(ctn_num), sum(units), vendor from inbound where (sap_code like '%584510753%' or item_num like '%584510753%') and date(date)='2024-11-11' group by sap_code, vendor ;
        """        
    },
    {
        "natural_language": "what did we receive on 2025-2-20?",
        "sql_query": """
        SELECT sap_code, item_num, sum(ctn_num), sum(units), vendor from inbound where date(date)='2025-02-20' group by sap_code, vendor ;
        """        
    },
    {
        "natural_language": "what items did we receive on 2025-2-20?",
        "sql_query": """
        SELECT sap_code, item_num, sum(ctn_num) , vendor from inbound where date(date)='2025-02-20' and qty>0 group by sap_code, vendor;
        """        
    },
    {
        "natural_language": "did we receive any containers on 2025-2-20?",
        "sql_query": """
        SELECT ref_num, vendor, date(date) from inbound where vendor = 'CONTAINER' and date(date)='2025-02-20' group by ref_num;
        """        
    },
    {
        "natural_language": "what is container numbers that  we have received on 2025-2-20?",
        "sql_query": """
        SELECT ref_num, date(date) from inbound where vendor = 'CONTAINER' and date(date)='2025-02-20' group by ref_num;
        """        
    },
    {
        "natural_language": "what is in the containers that  we have received on 2025-2-20?",
        "sql_query": """
        SELECT ref_num, sap_code, item_num, ctn_num, qty, pallet_num, vendor, date(date) from inbound where vendor = 'CONTAINER' and date(date)='2025-02-20';
        """        
    },    
    {
        "natural_language": "how many boxes of 584512679 did we receive on 2024-11-11?",
        "sql_query": """
        SELECT sap_code, item_num, sum(ctn_num), sum(qty) from inbound where date(date)='2024-11-11' and sap_code like '%584512679%' group by sap_code;
        """        
    },
    {
        "natural_language": "how many pallets of 584512679 did we receive on 2024-11-11?",
        "sql_query": """
        SELECT sap_code, item_num, sum(ctn_num), count(pallet_num) from inbound where date(date)='2024-11-11' and qty>0 and sap_code like '%584512679%' group by sap_code;
        """        
    },
    {
        "natural_language": "list outbound detail of 584512679 on 2024-11-11", 
        "sql_query": """
        SELECT sap_code, item_num, ctn_num, qty, pallet_num, location, lot_num, vendor, date from inbound where date(date)='2024-11-11' and sap_code like '%584512679%' and qty>0;
        """        
    },
    {
        "natural_language": "from where did we receive 584512679 and when", 
        "sql_query": """
        SELECT sap_code, item_num, vendor, date(date) from inbound where sap_code like '%584512679%' and qty>0;
        """        
    },
    {
        "natural_language": "do we have any expired items?",
        "sql_query": """
        SELECT item_code, item_num, description, units, exp_date, bin FROM inventory where units>0 and exp_date <= curdate() order by exp_date asc;
        """        
    },
    {
        "natural_language": "do we have any reserved items?",
        "sql_query": """
        SELECT i.item_num, i.sap_code, i.ctn_num, i.date, i.reserved_box, i.reserved_date, i.so_num FROM inbound i join warehouse w on FIND_IN_SET(w.so_num, i.so_num)>0 WHERE reserved_box > 0 AND (w.summary IS NULL OR w.summary = '') ;
        """
    },

    {
        "natural_language": "what shipped in today?",
        "sql_query": """
        SELECT sap_code, item_num, ctn_num, units, pallet_num, vendor FROM inbound WHERE date(date) = curdate();
        """
    },
    {
        "natural_language": "what shipped out yesterday?",
        "sql_query": """
        SELECT * FROM outbound WHERE date(shipped_date) = DATE_SUB(curdate(), INTERVAL 1 DAY) AND shipped='Y' AND confirmed = 'Y';
        """
    },
    {
        "natural_language": "List what we have received from FNS in January 2025",
        "sql_query": """
        SELECT item_num, sap_code,ctn_num, qty,pallet_num, location,vendor,date from inbound WHERE vendor = 'FNS' and date(date)>='2025-01-01' and date(date) <'2025-02-01';
        """
    },
    {
        "natural_language": "what items and how many boxes of them have we received in January 2025?",
        "sql_query": """
        SELECT item_num, sap_code,sum(ctn_num), sum(qty) from inbound WHERE date(date)>='2025-01-01' and date(date) <'2025-02-01' group by sap_code;
        """
    },
    {
        "natural_language": "what items and how many boxes of them have we received from FNS in January 2025?",
        "sql_query": """
        SELECT item_num, sap_code,sum(ctn_num), sum(qty), vendor from inbound WHERE date(date)>='2025-01-01' and date(date) <'2025-02-01' group by sap_code having vendor = 'FNS';
        """
    },
    {
        "natural_language": "what items and how many boxes of them have we unloaded from container in January 2025?",
        "sql_query": """
        SELECT item_num, sap_code,sum(ctn_num), sum(qty), vendor from inbound WHERE date(date)>='2025-01-01' and date(date) <'2025-02-01' group by sap_code having vendor = 'CONTAINER';
        """
    },
    {
        "natural_language": "what are container numbers of inbound shipments in January 2025?",
        "sql_query": """
        SELECT ref_num, vendor, date(date) from inbound WHERE date(date)>='2025-01-01' and date(date) <'2025-02-01' group by ref_num having vendor = 'CONTAINER';
        """
    }, 
    {
        "natural_language": "what was in container number BEAU6277469?",
        "sql_query": """
        SELECT item_num, sap_code, ctn_num, qty, pallet_num, location, ref_num, vendor, date(date) from inbound WHERE ref_num like '%BEAU6277469%' and vendor ='CONTAINER' and qty> 0;
        """
    },
    {
        "natural_language": "what was the  container number that we unloaded yesterday?",
        "sql_query": """
        SELECT distinct(ref_num) from inbound where vendor ='CONTAINER' and date(date)=DATE_SUB(curdate(), INTERVAL 1 DAY);
        """
    },
    {
        "natural_language": "how many containers did we unload in February 2025?",
        "sql_query": """
        SELECT COUNT(distinct(ref_num)), vendor from inbound WHERE vendor ='CONTAINER' and date(date)>='2025-02-01' and date(date) <'2025-03-01';
        """
    },
    {
        "natural_language": "how many containers did we receive in February 2025?",
        "sql_query": """
        SELECT COUNT(distinct(ref_num)), vendor from inbound WHERE vendor ='CONTAINER' and date(date)>='2025-02-01' and date(date) <'2025-03-01';
        """
    },
    {
        "natural_language": "what containers did we unload in February 2025?",
        "sql_query": """
        SELECT ref_num, vendor from inbound WHERE vendor ='CONTAINER' and date(date)>='2025-02-01' and date(date) <'2025-03-01' group by ref_num;
        """
    },
    {
        "natural_language": "who unloaded the container number BEAU6277469?",
        "sql_query": """
        SELECT distinct(username), ref_num, vendor, date(date) from inbound WHERE ref_num like '%BEAU6277469%' and vendor ='CONTAINER' and qty> 0 ;
        """
    },
    {
        "natural_language": "was any item returned in January 2025?",
        "sql_query": """
        SELECT item_num, sap_code, ctn_num, qty, ref_num, vendor, date(date) from inbound WHERE vendor ='RETURN' and date(date)>='2025-01-01' and date(date) <'2025-02-01' ;
        """
    },
    {
        "natural_language": "was any item returned?",
        "sql_query": """
        SELECT item_num, sap_code, ctn_num, qty, ref_num, vendor, date(date) from inbound WHERE vendor ='RETURN';
        """
    },
    {
        "natural_language": "what company returned?",
        "sql_query": """
        SELECT ref_num, vendor, date(date) from inbound WHERE vendor ='RETURN';
        """
    },
    {
        "natural_language": "what is sap code for mmlb8388?",
        "sql_query": """
        SELECT SAP, item_num, MASTER_QTY FROM products 
        WHERE item_num like '%mmlb8388%' ;
        """
    },
    {
        "natural_language": "what po is shipping out today via UPS or FEDEX?",
        "sql_query": """
        SELECT po_num, so_num, account, ctn_num, carrier FROM outbound where ETA = curdate() AND  shipped ='' AND (carrier like '%ups%' or carrier like '%fedex%');
        """
    },
    {
        "natural_language": "now, which crew finished those po's?",
        "sql_query": """

        SELECT o.po_num, o.so_num, o.account, o.ctn_num, o.carrier,w.crew FROM outbound o left join warehouse w on o.so_num = w.so_num where o.ETA = curdate() AND  o.shipped ='' AND (o.carrier like '%ups%' or o.carrier like '%fedex%');
        """
    
    },    

    {
        "natural_language": "List order details on po number 123456",
        "sql_query": """
        SELECT so_num, cust_po_num, customer, item_num, sap_code, order_qty FROM salesorder WHERE cust_po_num LIKE '%123456%';
        """
    },
    

    {
        "natural_language": "Who has po number 123456?",
        "sql_query": """
        SELECT so_num, po_num, crew, summary, process_start FROM warehouse WHERE po_num LIKE '%123456%' and summary='';
        """
    },
    {
        "natural_language": "Who finished po number 123456?",
        "sql_query": """
        SELECT so_num, po_num, crew, summary, process_start, process_end FROM warehouse WHERE po_num LIKE '%123456%' AND summary='Y';
        """
    },
    {
        "natural_language": "What is shipping out today?",
        "sql_query": """
        SELECT * FROM outbound WHERE ETD=CURDATE() AND (shipped!='Y' OR confirmed!='Y');
        """
    },
    {
        "natural_language": "Did OLD DOMINION pick up today?",
        "sql_query": """
        SELECT * FROM outbound WHERE DATE(shipped_date) = CURDATE() AND carrier  like '%OLD DOMINION%';
        """
    },
    {
        "natural_language": "What shipped out today?",
        "sql_query": """
        SELECT * FROM outbound WHERE DATE(shipped_date)=curdate() AND shipped ='Y' and confirmed ='Y';
        """
    },
    {
        "natural_language": "How many boxes did we ship out in March, 20205?",
        "sql_query": """
        SELECT sum(ctn_num) FROM outbound where date(shipped_date) >='2025-03-01' and date(shipped_date)<'2025-04-01';
        """        
    },
    
    {
        "natural_language": "How many tables are in the database?",
        "sql_query": """
        SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'creme';
        """
    },
    {
        "natural_language": "show pick ticket detail of po 4200070095 and expiration date of each item",
        "sql_query": """
        SELECT p.*,s.cust_po_num, s.customer, i.exp_date from picktix p  JOIN salesorder s on p.so_num = s.so_num and p.sap_code = s. sap_code JOIN inventory i on p.sap_code = i.item_code AND p.location = i.bin where s.cust_po_num ='4200070095';
        """
    },
    {
        "natural_language": "show pick ticket details of po number 215566675?",
        "sql_query": """
        SELECT p.*,s.cust_po_num, s.customer from picktix p  JOIN salesorder s on p.so_num = s.so_num and p.sap_code = s. sap_code where s.cust_po_num = '215566675'
        """
    },    
    {
        "natural_language": "Who is working on po number 215566675?",
        "sql_query": """
        SELECT po_num, so_num, crew, process_start, summary FROM warehouse WHERE po_num LIKE '%215566675%' AND process_start !='' AND summary='';
        """
    },
    {
        "natural_language": "Who has po 02224?",
        "sql_query": """
        SELECT so_num, po_num, crew, process_start FROM warehouse WHERE po_num LIKE '%02224%' and summary='';
        """,
    },
        {
        "natural_language": "What did T3 finish today?",
        "sql_query": """
        SELECT so_num, po_num, account, ctn_num, crew FROM warehouse WHERE crew like  LIKE '%T3%' and summary='Y' AND process_end=CURDATE();
        """,
    },
    {
        "natural_language": "Which product does not have UPC master or UPC unit?",
        "sql_query": """
        SELECT SAP,item_num, UPC_M, UPC_U FROM products WHERE UPC_M='' OR UPC_U='';
        """
    },
    {
        "natural_language": "Who from warehouse finished shipment that are scheduled to ship out via UPS or FEDEX today?",
        "sql_query": """
        SELECT outbound.account, outbound.so_num, outbound.po_num, outbound.carrier, outbound.ctn_num, outbound.ETD, outbound.shipped_date, warehouse.crew 
        FROM outbound 
        LEFT JOIN warehouse ON outbound.so_num=warehouse.so_num 
        WHERE outbound.ETD=CURDATE() AND (outbound.carrier like '%UPS%' OR outbound.carrier like '%FEDEX%');
        """
    },
        {
        "natural_language": "What is T3 working on?",
        "sql_query": """
        SELECT so_num, po_num, account, crew, ctn_num, process_start, summary FROM warehouse WHERE crew LIKE '%T3%' AND summary NOT IN ('Y','Ca') AND process_start!='' AND process_end='';
        """
    },
    {
        "natural_language": "When is po 0005355110 shipping out?",
        "sql_query": """
        SELECT po_num, so_num, account, ETD, carrier, bol, shipped 
        FROM outbound 
        WHERE po_num LIKE '%0005355110%';
        """
    },
    {
        "natural_language": "How many boxes did each crew finish in the month of May, 2024?",
        "sql_query": """
        SELECT SUM(ctn_num), crew 
        FROM warehouse 
        WHERE process_end>='2024-05-01' AND process_end<'2024-06-01' 
        GROUP BY crew;
        """
    },
    {
        "natural_language": "What are the items and the quantity of all POs finished in the warehouse in December?",
        "sql_query": """
        SELECT s.so_num, s.cust_po_num, s.customer, s.item_num, s.sap_code, s.order_qty 
        FROM salesorder s 
        LEFT JOIN warehouse w ON s.so_num = w.so_num 
        WHERE w.process_end>='2024-12-01' AND w.process_end<'2025-01-01' AND w.summary='Y';
        """
    },
    {
        "natural_language": "Did we receive any shipments today?",
        "sql_query": 
        "SELECT * FROM inbound WHERE date(`date`) = CURDATE() and qty>0;"
        
    },
    {
        "natural_language": "Did we receive any shipments on 2025-01-10?",
        "sql_query": """
        SELECT * FROM inbound WHERE date(`date`) = '2025-01-10' and qty>0;
        """
    },
    {
        "natural_language": "What are the top 5 most finished items and quantity in the warehouse for the month of December?",
        "sql_query": """
        SELECT s.item_num, s.sap_code, SUM(s.order_qty) AS quantity 
        FROM salesorder s 
        LEFT JOIN warehouse w ON s.so_num = w.so_num 
        WHERE w.process_end >= '2024-12-01' AND w.process_end <= '2024-12-31' 
        GROUP BY s.item_num, s.sap_code 
        ORDER BY quantity DESC 
        LIMIT 5;
        """
    },
    {
        "natural_language": "What was the top 10 most finished items in the warehouse for the months of October, November, December? Display item number, quantity, and the month.",
        "sql_query": """
        WITH ranked_items AS (
            SELECT
                s.sap_code,
                s.item_num,
                SUM(s.order_qty) AS quantity,
                EXTRACT(MONTH FROM w.process_end) AS month,
                RANK() OVER (PARTITION BY EXTRACT(MONTH FROM w.process_end) ORDER BY SUM(s.order_qty) DESC) AS rank
            FROM
                salesorder s
                LEFT JOIN warehouse w ON s.so_num = w.so_num
            WHERE
                w.process_end >= '2024-10-01'
                AND w.process_end <= '2024-12-31'
            GROUP BY
                s.item_num,
                s.sap_code,
                EXTRACT(MONTH FROM w.process_end)
        )
        SELECT
            sap_code,
            item_num,
            quantity,
            month
        FROM
            ranked_items
        WHERE
            rank <= 10
        ORDER BY
            month,
            rank;
        """
    },
        {
        "natural_language": "Do we have any repacking?",
        "sql_query": """
        SELECT r.so_num, s.cust_po_num AS cust_po_num, s.customer AS customer, r.from_sap_item, r.from_qty, r.to_sap_item, r.to_qty, r.status, w.summary, r.due_date, r.issue
        FROM repack AS r
        LEFT JOIN salesorder AS s ON s.so_num = r.so_num AND s.sap_code = r.to_sap_item
        LEFT JOIN warehouse AS w ON r.so_num = w.so_num
        WHERE (w.summary IS NULL OR w.summary = '') AND (r.`status` = 'Repacking' OR r.`status` = 'Order Received');
        """
    },
    {
        "natural_language": "Do we have any ETA items?",
        "sql_query": """
        SELECT s.so_num, s.customer, s.cust_po_num, s.item_num, s.sap_code, s.order_qty, p.qty FROM salesorder s
        LEFT JOIN warehouse w ON s.so_num = w.so_num
        LEFT JOIN (SELECT so_num, item_num, SUM(qty) AS qty FROM picktix WHERE `status` != 'Cancelled' GROUP BY so_num, item_num) p ON s.so_num = p.so_num AND s.item_num = p.item_num
        WHERE s.`status` = '' AND w.summary = '' AND w.routing_date != '' AND (s.order_qty>p.qty or p.qty IS NULL);
        """
    },
{
        "natural_language": "Is there any ETA items in po number oceana_65?",
        "sql_query": """    
        SELECT s.so_num, s.customer, s.cust_po_num, s.item_num, s.sap_code, s.order_qty, p.qty FROM salesorder s
        LEFT JOIN warehouse w ON s.so_num = w.so_num
        LEFT JOIN (SELECT so_num, item_num, SUM(qty) AS qty FROM picktix WHERE `status` != 'Cancelled' GROUP BY so_num, item_num) p ON s.so_num = p.so_num AND s.item_num = p.item_num
        WHERE s.`status` = '' AND w.summary = '' AND w.routing_date != '' AND (s.order_qty>p.qty OR p.qty IS NULL) AND s.cust_po_num like '%oceana_65%';
        """
    },
    {
        "natural_language": "Is there any ETA items in the po's that t1 is working on now?",
        "sql_query": """ 
        SELECT s.so_num, s.customer, s.cust_po_num, s.item_num, s.sap_code, s.order_qty, p.qty, w.summary FROM salesorder s
        LEFT JOIN warehouse w ON s.so_num = w.so_num
        LEFT JOIN (SELECT so_num, item_num, SUM(qty) AS qty FROM picktix WHERE `status` != 'Cancelled' GROUP BY so_num, item_num) p ON s.so_num = p.so_num AND s.item_num = p.item_num
        WHERE s.`status` = '' AND w.summary = ''  AND w.routing_date != '' AND w.process_start != '' AND w.crew='T1' AND (s.order_qty>p.qty or p.qty IS NULL);
        """
    },
    {
        "natural_language": "Do we have any repacking for PO number like 530160?",
        "sql_query": """
        SELECT r.so_num, s.cust_po_num AS cust_po_num, s.customer AS customer, r.from_sap_item, r.from_qty, r.to_sap_item, r.to_qty, r.status, w.summary, r.due_date, r.issue
        FROM repack AS r
        LEFT JOIN salesorder AS s ON s.so_num = r.so_num AND s.sap_code = r.to_sap_item
        LEFT JOIN warehouse AS w ON r.so_num = w.so_num
        WHERE (w.summary IS NULL OR w.summary = '') AND (r.`status` = 'Repacking' OR r.`status` = 'Order Received') AND s.cust_po_num LIKE '%530160%';
        """
    },
    {
        "natural_language": "Did we receive any ETA items yesterday?",
        "sql_query": """
        WITH eta_items AS (
        SELECT s.so_num, s.status, s.customer,s.cust_po_num, s.item_num,s.sap_code, s.order_qty, w.routing_date,p.qty 
            FROM salesorder s
            LEFT JOIN warehouse w ON s.so_num = w.so_num
            LEFT JOIN (SELECT so_num, item_num, SUM(qty) AS qty FROM picktix WHERE `status` != 'Cancelled' GROUP BY so_num, item_num) p ON s.so_num = p.so_num AND s.item_num = p.item_num
            WHERE s.`status` = '' AND w.summary = '' AND w.routing_date != '' AND (s.order_qty>p.qty OR p.qty IS NULL))
        SELECT 
            i.item_num,
            i.sap_code, 
            i.qty, 
            i.location, 
            i.pallet_num, 
            i.date,
            e.customer,
            e.cust_po_num,
            e.order_qty, 
            e.item_status
        FROM inbound i
        LEFT JOIN eta_items e ON i.sap_code = e.sap_code
        WHERE DATE(i.date) = CURDATE() - INTERVAL 1 DAY;
        """
    },
    {
        "natural_language": "Did we receive any ETA items on 2025-01-10?",
        "sql_query": """
        WITH eta_items AS (
        SELECT s.so_num, s.status, s.customer,s.cust_po_num, s.item_num,s.sap_code, s.order_qty, w.routing_date,p.qty 
            FROM salesorder s
            LEFT JOIN warehouse w ON s.so_num = w.so_num
            LEFT JOIN (SELECT so_num, item_num, SUM(qty) AS qty FROM picktix WHERE `status` != 'Cancelled' GROUP BY so_num, item_num) p ON s.so_num = p.so_num AND s.item_num = p.item_num
            WHERE s.`status` = '' AND w.summary = '' AND w.routing_date != '' AND (s.order_qty>p.qty OR p.qty IS NULL))
        SELECT 
            i.item_num,
            i.sap_code, 
            i.qty, 
            i.location, 
            i.pallet_num, 
            i.date,
            e.customer,
            e.cust_po_num,
            e.order_qty, 
            e.item_status
        FROM inbound i
        LEFT JOIN eta_items e ON i.sap_code = e.sap_code
        WHERE DATE(i.date) = '2025-01-10';
        """
         },
        {
        "natural_language": "is there any po's from sales order from 2025-01-01 that was not passed to warehouse?",
        "sql_query": """
        SELECT s.status, s.so_num, s.so_date, s.ship_date, s.customer, s.cust_po_num, sum(s.ctn_num) as ctn_num, w.rcvd_date
        FROM salesorder s
        LEFT JOIN warehouse AS w ON s.so_num = w.so_num
        
        WHERE (w.rcvd_date IS NULL OR w.rcvd_date = '') AND s.so_date>='2025-01-01' AND (s.status!='C' and s.status!='Ca')
        group by s.so_num;
        """
        },
        {
        "natural_language": "what is each crew working on now",
        "sql_query": """
        SELECT so_num, po_num, account, ctn_num, process_start, crew
        FROM warehouse
        WHERE summary='' AND process_start!='' AND process_end =''
        order by crew;
        """
        },
        {
        "natural_language": "what is item number and sap code that was received on 2024-11-11 from FNS?",
        "sql_query": """
        SELECT item_num, sap_code from outbound where date(date)='2024-11-11' and vendor = 'FNS';
        """
        },
        {
        "natural_language": "how many boxes of 584512679 did we receive on 2024-11-11 from FNS",
        "sql_query": """
        SELECT sum(ctn_num) from outbound where date(date)='2024-11-11' and vendor = 'FNS' and sap_code = '584512679';
        """
        },
        {
        "natural_language": "what is expiration date of 584512679 that we have received on 2024-11-11 from FNS",
        "sql_query": """
        SELECT distinct(lot_num) from outbound where date(date)='2024-11-11' and vendor = 'FNS' and sap_code = '584512679';
        """
        },
        {
        "natural_language": "what is pallet number of 584512679 that we have received on 2024-11-11 from FNS",
        "sql_query": """
        SELECT distinct(pallet_num) from outbound where date(date)='2024-11-11' and vendor = 'FNS' and sap_code = '584512679';
        """
        },
        {
        "natural_language": "how many pallets of 584512679 did we  receive on 2024-11-11 from FNS",
        "sql_query": """
        SELECT count(distinct(pallet_num)) from outbound where date(date)='2024-11-11' and vendor = 'FNS' and sap_code = '584512679';
        """
        },
        {
        "natural_language": "list all containers that we have received so far",
        "sql_query": """
        SELECT distinct(ref_num) from outbound where vendor = 'CONTAINER';
        """
        },
        {
        "natural_language": "show total number of boxes that was shipped out, that was received, and that was finished each month and year",
        "sql_query": """
        -- Step 1: Aggregate each table by year and month
WITH outbound_summary AS (
  SELECT
    YEAR(shipped_date) AS year,
    MONTH(shipped_date) AS month,
    SUM(ctn_num) AS outbound_boxes
  FROM outbound
  GROUP BY year, month
),
inbound_summary AS (
  SELECT
    YEAR(date) AS year,
    MONTH(date) AS month,
    SUM(ctn_num) AS inbound_boxes
  FROM inbound
  GROUP BY year, month
),
warehouse_summary AS (
  SELECT
    YEAR(process_end) AS year,
    MONTH(process_end) AS month,
    SUM(ctn_num) AS warehouse_boxes
  FROM warehouse
  GROUP BY year, month
)

-- Step 2: Full outer join simulation using UNION of LEFT JOINS
SELECT
  COALESCE(o.year, i.year, w.year) AS year,
  COALESCE(o.month, i.month, w.month) AS month,
  COALESCE(i.inbound_boxes, 0) AS inbound_boxes,
  COALESCE(o.outbound_boxes, 0) AS outbound_boxes,
  COALESCE(w.warehouse_boxes, 0) AS warehouse_boxes
FROM outbound_summary o
LEFT JOIN inbound_summary i ON o.year = i.year AND o.month = i.month
LEFT JOIN warehouse_summary w ON o.year = w.year AND o.month = w.month

UNION

SELECT
  COALESCE(i.year, w.year) AS year,
  COALESCE(i.month, w.month) AS month,
  COALESCE(i.inbound_boxes, 0) AS inbound_boxes,
  0 AS outbound_boxes,
  COALESCE(w.warehouse_boxes, 0) AS warehouse_boxes
FROM inbound_summary i
LEFT JOIN warehouse_summary w ON i.year = w.year AND i.month = w.month
WHERE NOT EXISTS (
  SELECT 1 FROM outbound_summary o
  WHERE o.year = i.year AND o.month = i.month
)

UNION

SELECT
  w.year,
  w.month,
  0 AS inbound_boxes,
  0 AS outbound_boxes,
  w.warehouse_boxes
FROM warehouse_summary w
WHERE NOT EXISTS (
  SELECT 1 FROM outbound_summary o
  WHERE o.year = w.year AND o.month = w.month
)
AND NOT EXISTS (
  SELECT 1 FROM inbound_summary i
  WHERE i.year = w.year AND i.month = w.month
)

ORDER BY year, month;
        """
        },

        {
        "natural_language": "warehouse activity report according to number of boxes each month of year",
        "sql_query": """
        -- Step 1: Aggregate each table by year and month
WITH outbound_summary AS (
  SELECT
    YEAR(shipped_date) AS year,
    MONTH(shipped_date) AS month,
    SUM(ctn_num) AS outbound_boxes
  FROM outbound
  GROUP BY year, month
),
inbound_summary AS (
  SELECT
    YEAR(date) AS year,
    MONTH(date) AS month,
    SUM(ctn_num) AS inbound_boxes
  FROM inbound
  GROUP BY year, month
),
warehouse_summary AS (
  SELECT
    YEAR(process_end) AS year,
    MONTH(process_end) AS month,
    SUM(ctn_num) AS warehouse_boxes
  FROM warehouse
  GROUP BY year, month
)

-- Step 2: Full outer join simulation using UNION of LEFT JOINS
SELECT
  COALESCE(o.year, i.year, w.year) AS year,
  COALESCE(o.month, i.month, w.month) AS month,
  COALESCE(i.inbound_boxes, 0) AS inbound_boxes,
  COALESCE(o.outbound_boxes, 0) AS outbound_boxes,
  COALESCE(w.warehouse_boxes, 0) AS warehouse_boxes
FROM outbound_summary o
LEFT JOIN inbound_summary i ON o.year = i.year AND o.month = i.month
LEFT JOIN warehouse_summary w ON o.year = w.year AND o.month = w.month

UNION

SELECT
  COALESCE(i.year, w.year) AS year,
  COALESCE(i.month, w.month) AS month,
  COALESCE(i.inbound_boxes, 0) AS inbound_boxes,
  0 AS outbound_boxes,
  COALESCE(w.warehouse_boxes, 0) AS warehouse_boxes
FROM inbound_summary i
LEFT JOIN warehouse_summary w ON i.year = w.year AND i.month = w.month
WHERE NOT EXISTS (
  SELECT 1 FROM outbound_summary o
  WHERE o.year = i.year AND o.month = i.month
)

UNION

SELECT
  w.year,
  w.month,
  0 AS inbound_boxes,
  0 AS outbound_boxes,
  w.warehouse_boxes
FROM warehouse_summary w
WHERE NOT EXISTS (
  SELECT 1 FROM outbound_summary o
  WHERE o.year = w.year AND o.month = w.month
)
AND NOT EXISTS (
  SELECT 1 FROM inbound_summary i
  WHERE i.year = w.year AND i.month = w.month
)

ORDER BY year, month;
        """
        },
        {
        "natural_language": "how many containers did we unload each month",
        "sql_query": """
        SELECT YEAR(date) AS year, MONTH(date) AS month, count(distinct(ref_num)) AS num_containers FROM inbound WHERE date(date) BETWEEN '2023-01-01' AND CURRENT_DATE() and vendor ='CONTAINER' GROUP BY year, month;
        """
        },
        {
        "natural_language": "list salesorder details of orders that was finished in warehouse but has not shipped out yet",
        "sql_query": """
SELECT s.so_num, s.customer, s.cust_po_num, s.item_num, s.sap_code, s.order_qty FROM `salesorder` s 
join warehouse w on s.so_num = w.so_num 
join outbound o on (s.so_num = o.so_num OR s.cust_po_num =o.po_num)
WHERE s.`status` NOT IN ('C','Ca') AND w.summary ='Y' AND w.routing_date !='' and (o.shipped_date = '' or o.shipped_date IS NULL) and o.shipped = ''
        """
        },
    {
        "natural_language": "Are we expecting any shipment today from FNS?",
        "sql_query": """
        SELECT * FROM sto_inbound WHERE rdd = CURDATE();
        """
    },
    {
        "natural_language": "Are we expecting any STO from FNS today?",
        "sql_query": """
        SELECT * FROM sto_inbound WHERE rdd = curdate();
        """
    },
    {
        "natural_language": "how many boxes were sold each month of the year?",
        "sql_query": """
        SELECT YEAR(so_date) AS year, MONTH(so_date) AS month, SUM(ctn_num) AS total_boxes_sold, SUM(units) AS total_units_sold FROM salesorder GROUP BY year, month ORDER BY year, month;
        """
    },
    {
        "natural_language": "how many boxes were finished each month of the year in warehouse?",
        "sql_query": """
        SELECT YEAR(process_end) AS year, MONTH(process_end) AS month, SUM(ctn_num) AS total_boxes_finished FROM warehouse GROUP BY year, month ORDER BY year, month;
        """
    },
    {
        "natural_language": "how many boxes were processed each month of the year in warehouse?",
        "sql_query": """
        SELECT YEAR(process_end) AS year, MONTH(process_end) AS month, SUM(ctn_num) AS total_boxes_finished FROM warehouse GROUP BY year, month ORDER BY year, month;
        """
    },
    {
        "natural_language": "how many boxes were shipped out each month of the year?",
        "sql_query": """
        SELECT YEAR(shipped_date) AS year, MONTH(shipped_date) AS month, SUM(ctn_num) AS total_boxes_outbound FROM outbound GROUP BY year, month ORDER BY year, month;
        """
    },
    {
        "natural_language": "how many boxes did we ship out each month of the year?",
        "sql_query": """
        SELECT YEAR(shipped_date) AS year, MONTH(shipped_date) AS month, SUM(ctn_num) AS total_boxes_outbound FROM outbound GROUP BY year, month ORDER BY year, month;
        """
    },
    {
        "natural_language": "how many boxes did we receive each month of the year?",
        "sql_query": """
        SELECT YEAR(date) AS year, MONTH(so_date) AS month, SUM(ctn_num) AS total_boxes_inbound FROM inbound GROUP BY year, month ORDER BY year, month;
        """,

        "natural_language": "Can you break it down by vendors?",
        "sql_query": """
        SELECT YEAR(date) AS year, MONTH(date) AS month, vendor, SUM(ctn_num) AS total_boxes_inbound FROM inbound GROUP BY year, month ORDER BY year, month,vendor;
        """
    },
    {
        "natural_language": "what work orders do we have in warehouse that has not been started?",
        "sql_query": """
        SELECT * from warehouse where routing_date != '' and summary ='' and process_start = '';
        """
    },
    {
        "natural_language": "list work orders in warehouse that the fulfillment process has not begun yet",
        "sql_query": """
        SELECT so_num, po_num, account, ctn_num from warehouse where routing_date != '' and summary ='' and process_start = '';
        """
    },
    {
        "natural_language": "what is the name of account or customer in repack?",
        "sql_query": """
        SELECT DISTINCT r.so_num, s.customer FROM repack AS r JOIN salesorder AS s ON s.so_num = r.so_num;
        """
    },
    {
        "natural_language": "where was the repacking item/product picked from for SO 1234?",
        "sql_query": """
        SELECT  d.so_num, s.customer, s.cust_po_num, d.from_sap_item, d.from_qty, d.location, d.picked_qty, d.picked_date FROM delivery_order AS d JOIN salesorder AS s ON s.so_num = d.so_num where d.so_num = '1234';
        """
    },
    {
        "natural_language": "can you show repacking detail for product 12345",
        "sql_query": 
        """
        SELECT  r.so_num, s.customer, s.cust_po_num, d.from_sap_item, r.from_qty, d.location, d.picked_qty, d.picked_date, r.to_sap_item, r.to_qty, r.issue, w.summary FROM repack r 
        JOIN salesorder AS s ON s.so_num = d.so_num and s.sap_code = r.to_sap_code 
        Join delivery_order d on r.so_num = d.so_num and r.from_sap_item = d.from_sap_item 
        Join warehouse w on r.so_num = w.so_num
        where r.from_sap_item = '1234' or r.to_sap_item = '1234' 
        group by r.so_num, s.customer, s.cust_po_num, d.from_sap_item, r.from_qty, d.location, d.picked_qty, d.picked_date, r.to_sap_item, r.to_qty, r.issue, w.summary;
        """
    },
    {
        "natural_language": "what is the name of account or customer of order 123456?",
        "sql_query": """
        SELECT DISTINCT  customer FROM salesorder where so_num = '123456';
        """
    },

    {
        "natural_language": "List pick ticket details of purchase order 11319997",
        "sql_query": """
        SELECT p.so_num, p.sap_code, p.item_num, , p.status, p.date, p.pt_num, p.location, p.qty FROM picktix p join salesorder s on p.so_num = s.so_num and s.sap_code = p.sap_code  where s.status != 'Ca' and s.cust_po_num = '11319997';
        """
    },

    {
        "natural_language": "List products and quantity that was shipped out in first quarter of 2025",
        "sql_query": """
        SELECT s.item_num, s.sap_code, p.description, sum(s.order_qty) as total_qty FROM salesorder s join (select so_num from outbound WHERE shipped = 'Y' AND date(shipped_date)>= '2025-01-01' AND date(shipped_date) < '2025-04-01' ) o on s.so_num = o.so_num left join products p on s.sap_code = p.SAP where s.status !='Ca' group by sap_code
        order by total_qty desc;;
        """
    },

    {
        "natural_language": "List products and quantity that was finished in first quarter of 2025",
        "sql_query": """
        SELECT s.item_num, s.sap_code, p.description, sum(s.order_qty) as total_qty FROM salesorder s join (select so_num from warehouse WHERE summary = 'Y' AND date(process_end)>= '2025-01-01' AND date(process_end) < '2025-04-01' ) o on s.so_num = o.so_num left join products p on s.sap_code = p.SAP where s.status !='Ca' group by sap_code
        order by total_qty desc;;
        """
    },
    {
        "natural_language": "show pick ticket details of what T1 is working on now",
        "sql_query": """
        SELECT p.*, s.cust_po_num, s.customer 
        FROM warehouse w 
        JOIN picktix p ON w.so_num = p.so_num 
        JOIN salesorder s ON p.so_num = s.so_num AND p.sap_code = s.sap_code 
        WHERE w.crew = 'T1' 
        AND w.summary NOT IN ('Y', 'Ca') 
        AND w.process_start != '' 
        AND w.routing_date != ''
        """
    },
    {
        "natural_language": "what are the products in inventory that have not been sold in last 6 months?",
        "sql_query":"""
        SELECT
        i.item_code, i.description, i.units, i.bin,
        ls.last_salesorder_date
        FROM inventory i
        LEFT JOIN (
        SELECT sap_code, MAX(so_date) AS last_salesorder_date
        FROM salesorder
        GROUP BY sap_code
        ) ls ON ls.sap_code = i.item_code
        WHERE i.units > 0
        AND (ls.last_salesorder_date < DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
            OR ls.last_salesorder_date IS NULL);
            
        """
    },
    {
        "natural_language": "is there anything missing for po 1234?",
        "sql_query": """
        SELECT p.*, s.cust_po_num, s.customer FROM picktix p join salesorder s on p.so_num = s.so_num WHERE (missing_ctn_num !='0' or missing_units != '0') and p.status = 'Active' and s.cust_po_num = '1234' group by p.so_num, p.sap_code;
        """
    },
    {
        "natural_language": "is there anything missing for po's that are currently being worked on?",
        "sql_query": """
        SELECT p.*, s.cust_po_num , s.customer, w.process_start, w.summary, w.crew FROM picktix p join salesorder s on p.so_num = s.so_num join warehouse w on p.so_num = w.so_num WHERE (missing_ctn_num !='0' or missing_units != '0') and p.status = 'Active' and w.process_start !='' and w.summary ='' group by p.so_num, p.sap_code;
        """
    },
    {
        "natural_language": "what is each crew working on?",
        "sql_query": """
        SELECT * FROM warehouse where process_start !='' and summary ='' and crew !='' order by crew;
        """
    },
    {
        "natural_language": "show total units for each product in inventory",
        "sql_query": """
        SELECT item_code, item_num, description, SUM(units) as total_units 
        FROM inventory 
        WHERE date = (SELECT MAX(date) FROM inventory) AND units > 0
        GROUP BY item_code, item_num, description
        ORDER BY total_units DESC;
        """
    },
    {
        "natural_language": "how many pick tickets are there for each sales order?",
        "sql_query": """
        SELECT p.so_num, COUNT(*) as ticket_count, SUM(p.qty) as total_qty
        FROM picktix p
        WHERE p.status = 'Active'
        GROUP BY p.so_num
        ORDER BY ticket_count DESC;
        """
    },
    {
        "natural_language": "show each product with how many different locations it's stored in",
        "sql_query": """
        SELECT item_code, item_num, description, 
               COUNT(DISTINCT bin) as location_count,
               SUM(units) as total_units
        FROM inventory 
        WHERE date = (SELECT MAX(date) FROM inventory) AND units > 0
        GROUP BY item_code, item_num, description
        HAVING COUNT(DISTINCT bin) > 1
        ORDER BY location_count DESC;
        """
    },
    {
        "natural_language": "list products with their total quantity ordered across all sales orders",
        "sql_query": """
        SELECT sap_code, item_num, 
               COUNT(DISTINCT so_num) as order_count,
               SUM(order_qty) as total_ordered
        FROM salesorder 
        WHERE status != 'Ca'
        GROUP BY sap_code, item_num
        ORDER BY total_ordered DESC;
        """
    },
    {
        "natural_language": "show pick ticket summary for each active sales order with customer info",
        "sql_query": """
        SELECT p.so_num, s.customer, s.cust_po_num,
               COUNT(DISTINCT p.sap_code) as unique_products,
               SUM(p.qty) as total_qty
        FROM picktix p
        JOIN salesorder s ON p.so_num = s.so_num
        WHERE p.status = 'Active'
        GROUP BY p.so_num, s.customer, s.cust_po_num;
        """
    },
    {
        "natural_language": "what vendors sent us the most boxes?",
        "sql_query": """
        SELECT vendor, 
               COUNT(DISTINCT ref_num) as shipment_count,
               SUM(ctn_num) as total_boxes
        FROM inbound
        WHERE vendor NOT IN ('RETURN', 'CONTAINER')
        GROUP BY vendor
        ORDER BY total_boxes DESC;
        """
    },
    {
        "natural_language": "show monthly receiving totals for this year",
        "sql_query": """
        SELECT YEAR(date) as year, 
               MONTH(date) as month,
               COUNT(*) as receipt_count,
               SUM(ctn_num) as total_boxes,
               SUM(units) as total_units
        FROM inbound
        WHERE YEAR(date) = YEAR(CURDATE())
        GROUP BY YEAR(date), MONTH(date)
        ORDER BY year, month;
        """
    },
    {
        "natural_language": "which products appear in the most sales orders?",
        "sql_query": """
        SELECT sap_code, item_num,
               COUNT(DISTINCT so_num) as order_count,
               SUM(order_qty) as total_qty
        FROM salesorder
        WHERE status != 'Ca'
        GROUP BY sap_code, item_num
        ORDER BY order_count DESC
        LIMIT 20;
        """
    },

]







