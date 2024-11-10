module HelloWorld(
    input clk,                            // Main clock input
    output [0 : 6] display0_Output,       // Combined outputs for SSD 0
    output [0 : 6] display1_Output,       // Combined outputs for SSD 1
    output [0 : 6] display2_Output,       // Combined outputs for SSD 2
    output [0 : 6] display3_Output,       // Combined outputs for SSD 3
    output [0 : 6] display4_Output,       // Combined outputs for SSD 4
    output [0 : 6] display5_Output        // Combined outputs for SSD 5
);

    wire clk_slow;                        // Slow clock for display toggling
    reg display_toggle = 0;               // Toggles between HELLO and WORLD

    // Clock divider instance
    ClockDivider clk_divider(
        .cin(clk),
        .cout(clk_slow)
    );

    // Toggle the display every 2 seconds
    always @(posedge clk_slow) begin
        display_toggle <= ~display_toggle;
    end

    /* 
        HELLO and WORLD segments combined based on display_toggle
    */

    // SSD 0 (H, W part 1)
    assign display0_Output = (display_toggle == 0) ? 7'b1001000 : 7'b1100001;

    // SSD 1 (E, W part 2)
    assign display1_Output = (display_toggle == 0) ? 7'b0110000 : 7'b1000011;

    // SSD 2 (L, O)
    assign display2_Output = (display_toggle == 0) ? 7'b1110001 : 7'b0000001;

    // SSD 3 (L, R)
    assign display3_Output = (display_toggle == 0) ? 7'b1110001 : 7'b1111010;

    // SSD 4 (O, L)
    assign display4_Output = (display_toggle == 0) ? 7'b0000001 : 7'b1110001;

    // SSD 5 (blank, D)
    assign display5_Output = (display_toggle == 0) ? 7'b1111111 : 7'b1000010;

endmodule