gsap.from("#nav",{
    y:-30,
    duration:0.8,
    delay:0.3,
    opacity:0
})

gsap.from("#platform",{
    y:100,
    duration:0.6,
    opacity:0
})

gsap.from("#gola",{
    y:-600,
    duration:1.2,
    delay:0.3,
    opacity:0
})

gsap.from("#page1>h1",{
    y:100,
    duration:1.2,
    delay:1.3,
    opacity:0,
    onStart:function(){
        $('#page1>h1').textillate({ in: { effect: 'fadeInUp' } });   
    }
})


var tl = gsap.timeline({
    scrollTrigger:{
        trigger:"#page1",
        scroller:"body",
        // markers:true,
        start:"top -5%",
        scrub:3
    }
})

var tl2 = gsap.timeline({
    scrollTrigger:{
        trigger:"#page1",
        scroller:"body",
        // markers:true,
        start:"top -60%",
        scrub:1,
        duration:0.1
    }
})

tl.to("#gola",{
    x:950,
    top:"50vh",
    rotate:360
},"anim1")

tl.to("#platform",{
    rotate:15
},"anim1")

tl2.to("#platform",{
    rotate:0
})

tl.to("#page2-in>h1",{
    delay:0.2,
    onStart:function(){
        $('#page2-in>h1').textillate({ in: { effect: 'fadeInUp' } });   
    }
},"anim1")

tl.to("#page2-content>p",{
    delay:0.3,
    onStart:function(){
        $("#age2-content>p").textillate({ in: { effect: 'fadeInUp' } });   
    }
}, "anim1")

