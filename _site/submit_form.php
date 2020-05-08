<?php
phpinfo();

if(isset($_POST['submit'])){
    $to = "viha4393@colorado.edu"; // this is your Email address
    $from = $_POST['email']; // this is the sender's Email address
    $name = $_POST['name'];
    $subject = "Form submission";
    // $subject2 = "Copy of your form submission";
    $message = $_POST['message'];
    // $message2 = $_POST['message'];

    $headers = "From:";
    // $headers2 = "From:";
    mail($to,$subject,$message,$headers);
    // mail($from,$subject2,$message2,$headers2); // sends a copy of the message to the sender
    echo "Mail Sent. Thank you ";
    }
?>
