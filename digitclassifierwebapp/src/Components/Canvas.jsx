import React, { Component } from 'react';
import CanvasDraw from "react-canvas-draw";
import imageSrc from "./imgSrc/black_bg.jpg";
import classNames from "./CSS/index.css";

export default class Canvas extends Component {
    
    state = {
        brushColor:"rgb(255,255,255)",
        width: 500,
        height: 500,
        brushRadius: 10,
        lazyRadius: 12,
        result: 0,
        accuracy: 0.0
    };

    style = {
        fontSize: 20,
        width: 100
    }

    RenderBrushSize(){
        return (
            <div className="tools">
                <label>Brush-Radius:</label>
                    <input
                    type="number"
                    value={this.state.brushRadius}
                    onChange={e =>
                        this.setState({ brushRadius: parseInt(e.target.value, 10) })
                    }
                />
            </div>
        );
    }

    RenderButtons(){
        return (
            <div id="middle">

                <div className="button-padding">
                    <button style={this.style} className="btn btn-primary"
                        onClick={() => {
                        localStorage.setItem(
                            "savedDrawing",
                            this.ProcessDrawing()
                        );
                        }}> Save </button> 
                </div>

                <div className="button-padding">
                    <button style={this.style} className="btn btn-primary"
                        onClick={() => {
                            // revert back to original state
                            this.saveableCanvas.clear();
                            this.setState({
                                result: 0,
                                accuracy: 0.0
                            });
                        }}> Clear </button>
                </div>
                
                <div className="button-padding">
                    <button style={this.style} className="btn btn-primary"
                        onClick={() => {
                        this.saveableCanvas.undo();
                        }}> Undo </button>
                </div>
            </div>
        );
    }

    RenderCanvas(){
        //Utilizing third party library to render canvas
        return (
            <div id="left" width={this.state.width}>
                <CanvasDraw 
                    ref={canvasDraw => (this.saveableCanvas = canvasDraw)}
                    brushColor={this.state.brushColor}
                    imgSrc={imageSrc}
                    brushRadius={this.state.brushRadius}
                    lazyRadius={this.state.lazyRadius}
                    canvasWidth={this.state.width}
                    canvasHeight={this.state.height}
                />
            </div>
        );
    }

    RenderResults(){
        return (
            <div id="right">
                <span className="badge badge-primary m-2">Result: {this.state.result}</span>
                <span className="badge badge-primary">Accuracy: {this.state.accuracy}%</span>
            </div>
        );
    }

    ProcessDrawing(){
        //Process the drawing to back end and return output
        //Passing as a string, as i cant pass images directly to the server side
        const imageUri = this.saveableCanvas.canvasContainer.children[1].toDataURL("image/png");
        if (imageUri) {

            const url = "http://localhost:5000/process_image";
            fetch(url, {
                method:"POST",
                cache: "no-store",
                //Headers are required to bypass the CORS block, as the applocation is communicating from 2 different ports
                headers:{
                    'Accept': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Origin, X-Requested-With, Content-Type, Accept, Authorization',
                    'Access-Control-Request-Method': 'GET, POST, DELETE, PUT, OPTIONS'
                },
                body: JSON.stringify(imageUri)
                }
            ).then(response => {
                console.log(response);
                return response.json();
            })
            .then(contents => {
                console.log(contents);
                //Applying the returned values to the states 
                this.setState({
                    result: contents["Result"],
                    accuracy: contents["Accuracy"]
                });
            })
            .catch(() => console.log("Can’t access " + url + " response. Blocked by browser?"));
        }
    }

    render() { 
        return ( 
            <React.Fragment>
                <div >
                    <h1>CNN MNSIT Digit classifier by Ashraf Rahman</h1>
                </div>

                {this.RenderBrushSize()}
                
                <div id="container">
                    {this.RenderCanvas()}

                    {this.RenderButtons()}

                    {this.RenderResults()}
                </div>

            </React.Fragment>
         );
    }
}