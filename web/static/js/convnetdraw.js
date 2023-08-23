var convnetdraw = { REVISION: 'ALPHA' };
(function (global) {
    "use strict";

    var drawing = function (id) {
        // Set up our canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = document.getElementById(id).clientWidth;
        this.canvas.height = document.getElementById(id).clientHeight;

        var content = document.getElementById(id);
        content.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');

        this.ratio1 = document.querySelector('#ratio1');
        this.ratio2 = document.querySelector('#ratio2');

        this.zoomx = document.querySelector('#zoomx');
        this.zoomz = document.querySelector('#zoomz');
        this.zoomy = document.querySelector('#zoomy');


    }

    drawing.prototype = {
        drawItem: function (title, dim0, dim1, dim2, color) {
            var actualX = dim2 * this.zoomx.value / 20;
            var actualZ = dim1 * this.zoomz.value * 3;
            var actualY = dim0 * this.zoomy.value * 3;

            // draw the cube
            this.drawCube(
                this.offset,
                this.canvas.clientHeight / 2 + dim1 * this.zoomy.value / 2,
                Number(actualX),
                Number(actualZ),
                Number(actualY),
                color
            );
            var text1 = `${title}`;
            var text = `${dim0}x${dim1}x${dim2}`;
            var text1Width = this.ctx.measureText(text1).width;
            var textWidth = this.ctx.measureText(text).width;
            this.ctx.fillStyle = "rgb(26, 25, 25)";
            this.ctx.fillText(text,
                this.offset
                + actualY * this.ratio1.value / 1000
                + (actualX - textWidth) / 2 - 10,
                this.canvas.clientHeight / 2 - dim0 * (0.5 + this.ratio2.value / 1000) * this.zoomy.value + 80);

            this.ctx.fillText(text1,
                this.offset
                + actualY * this.ratio1.value / 1000
                + (actualX - text1Width) / 2 - 10,
                this.canvas.clientHeight / 2 - dim0 * (0.5 + this.ratio2.value / 1000) * this.zoomy.value + 60);

            this.offset += actualX + actualY * this.ratio1.value / 1000 + 5;
        },

        drawItem1: function (title, dim0, dim1, dim2, color) {
            var actualX = dim2 * this.zoomx.value / 50;
            var actualZ = dim1 * this.zoomz.value * 4;
            var actualY = dim0 * this.zoomy.value * 4;

            // draw the cube
            this.drawCube(
                this.offset,
                this.canvas.clientHeight / 2 + dim1 * this.zoomy.value / 2,
                Number(actualX),
                Number(actualZ),
                Number(actualY),
                color
            );
            var text1 = `${title}`;
            var text1Width = this.ctx.measureText(text1).width;
            this.ctx.fillStyle = "rgb(26, 25, 25)";
            this.ctx.fillText(text1,
                this.offset
                + actualY * this.ratio1.value / 1000
                + (actualX - text1Width) / 2 - 20,
                this.canvas.clientHeight / 2 - dim0 * (0.5 + this.ratio2.value / 1000) * this.zoomy.value + 68);

            this.offset += actualX + actualY * this.ratio1.value / 1000 + 5;
        },

        save: function () {
            var win = window.open();
            var dataUrl = this.canvas.toDataURL("image/png");
            win.document.write("<img src='" + dataUrl + "'/>");
        },

        draw: function (text, ofs = 10) {

            this.offset = ofs;

            // clear the canvas
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);


            for (var i = 0; i < text.length; i++) {
                try {

                    eval("this." + text[i]);
                } catch (e) {
                    var myRegexp = /(.*)\((.*),(.*),(.*)\)/g;
                    var match = myRegexp.exec(text[i]);
                    if (match != null) {
                        this.generic(match[1], match[2], match[3], match[4]);
                    }

                }


            }

            // requestAnimationFrame(this.draw.bind(this, text));
        },

        input: function (dim0, dim1, dim2) {
            this.drawItem1("input", dim0, dim1, dim2, "#00c8fa");
        },

        conv: function (dim0, dim1, dim2) {
            this.drawItem("conv", dim0, dim1, dim2, "#5bc3c4");
        },

        relu: function (dim0, dim1, dim2) {
            this.drawItem1("relu", dim0, dim1, dim2, "#ffc33f");
        },

        pool: function (dim0, dim1, dim2) {
            this.drawItem1("pool", dim0, dim1, dim2, "#f9e3b4");
        },

        fullyconn: function (dim0, dim1, dim2) {
            this.drawItem1("fullyconn", dim0, dim1, dim2, "#fe4d66");
        },

        softmax: function (dim0, dim1, dim2) {
            this.drawItem1("softmax", dim0, dim1, dim2, "#00c8fa");
        },

        generic: function (text, dim0, dim1, dim2) {
            this.drawItem("conv", dim0, dim1, dim2, "#00c8fa");
        },

        // Colour adjustment function
        // Nicked from http://stackoverflow.com/questions/5560248
        shadeColor: function (color, percent) {


            color = color.substr(1);
            var num = parseInt(color, 16),
                amt = Math.round(2.55 * percent),
                R = (num >> 16) + amt,
                G = (num >> 8 & 0x00FF) + amt,
                B = (num & 0x0000FF) + amt;
            return '#' + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 + (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 + (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
        },

        // Draw a cube to the specified specs
        drawCube: function (x, y, wx, wy, h, color) {
            this.ctx.beginPath();
            this.ctx.moveTo(x + wx, y);
            this.ctx.lineTo(x, y);
            this.ctx.lineTo(x, y - h);
            this.ctx.lineTo(x + wx, y - h * 1);
            this.ctx.closePath();
            this.ctx.fillStyle = this.shadeColor(color, -10);
            this.ctx.strokeStyle = color;
            this.ctx.stroke();
            this.ctx.fill();

            this.ctx.beginPath();
            this.ctx.moveTo(x + wx, y);
            this.ctx.lineTo(x + wx + wy * this.ratio1.value / 1000, y - wy * this.ratio2.value / 1000);
            this.ctx.lineTo(x + wx + wy * this.ratio1.value / 1000, y - h - wy * this.ratio2.value / 1000);
            this.ctx.lineTo(x + wx, y - h * 1);
            this.ctx.closePath();
            this.ctx.fillStyle = this.shadeColor(color, 10);
            this.ctx.strokeStyle = this.shadeColor(color, 50);
            this.ctx.stroke();
            this.ctx.fill();

            this.ctx.beginPath();
            this.ctx.moveTo(x + wx, y - h);
            this.ctx.lineTo(x, y - h);
            this.ctx.lineTo(x + wy * this.ratio1.value / 1000, y - h - (wy * this.ratio2.value / 1000));
            this.ctx.lineTo(x + wx + wy * this.ratio1.value / 1000, y - h - wy * this.ratio2.value / 1000);
            this.ctx.closePath();
            this.ctx.fillStyle = this.shadeColor(color, 20);
            this.ctx.strokeStyle = this.shadeColor(color, 60);
            this.ctx.stroke();
            this.ctx.fill();


            this.ctx.beginPath();
            this.ctx.moveTo(x + wx, y);
            this.ctx.lineTo(x, y);
            this.ctx.lineTo(x, y - h);
            this.ctx.lineTo(x + wx, y - h * 1);
            this.ctx.closePath();
            this.ctx.strokeStyle = "black";
            this.ctx.stroke();

            this.ctx.beginPath();
            this.ctx.moveTo(x + wx, y);
            this.ctx.lineTo(x + wx + wy * this.ratio1.value / 1000, y - wy * this.ratio2.value / 1000);
            this.ctx.lineTo(x + wx + wy * this.ratio1.value / 1000, y - h - wy * this.ratio2.value / 1000);
            this.ctx.lineTo(x + wx, y - h * 1);
            this.ctx.closePath();
            this.ctx.strokeStyle = "black";
            this.ctx.stroke();

            this.ctx.beginPath();
            this.ctx.moveTo(x + wx, y - h);
            this.ctx.lineTo(x, y - h);
            this.ctx.lineTo(x + wy * this.ratio1.value / 1000, y - h - (wy * this.ratio2.value / 1000));
            this.ctx.lineTo(x + wx + wy * this.ratio1.value / 1000, y - h - wy * this.ratio2.value / 1000);
            this.ctx.closePath();
            this.ctx.strokeStyle = "black";
            this.ctx.stroke();
        }
    }

    global.drawing = drawing;

})(convnetdraw);