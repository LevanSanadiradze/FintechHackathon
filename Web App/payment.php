<?php
$data = array('ServiceId' => '1',
              'UserIdentifier' => '1',
              'CardPan' => 'a9898989898aas42664e07b93cce044f',
              'Amount' => 4.2,
              'Currency' => 'GEL');

$data_string = json_encode($data);    

$ch = curl_init();

curl_setopt($ch, CURLOPT_URL,"https://api.fintech.ge/api/Transfers/Payment");
curl_setopt($ch, CURLOPT_CUSTOMREQUEST, "POST");                                                                     
curl_setopt($ch, CURLOPT_POSTFIELDS, $data_string);                                                                  
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);                                                                      
curl_setopt($ch, CURLOPT_HTTPHEADER, array(                                                                          
    'Content-Type: application/json',                                                                                
    'Content-Length: ' . strlen($data_string)));

$server_output = curl_exec($ch);

curl_close ($ch);

var_dump($server_output);

?>